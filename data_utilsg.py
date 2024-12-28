import copy
import json
import pickle
import random
import re
import gc
import sys
import multiprocessing
import shutil
#import resource
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
import enum
import os, psutil
import sympy
import networkx as nx 

# An exception to limit the maximum number of allowed transformations 
class NbTranformationException(Exception):
    pass

# An exception to limit the maximum number of read-write accesses. 
class NbAccessException(Exception):
    pass

# An exception to limit the maximum number of nested loops. Currently set to 5.
class LoopsDepthException(Exception):
    pass

# Maximum sequence of transformations (reversal, interchange and skewing) allowed. Currently set to 4 
MAX_NUM_TRANSFORMATIONS = 4

# Maximum size of the tags vector representing each transformation
MAX_TAGS = 16

# Maximum depth of a loop nest for each computation
MAX_DEPTH = 5

# Maximum length of expressions in the dataset
MAX_EXPR_LEN = 66

def get_representation_template(program_dict, train_device="cpu"):
    graph = {
        "nodes": {},  # Holds nodes with attributes
        "edges": [],  # Holds edges representing dependencies and hierarchy
    }

    program_json = program_dict["program_annotation"]
    computations_dict = program_json["computations"]
    ordered_comp_list = sorted(
        computations_dict.keys(),
        key=lambda x: computations_dict[x]["absolute_order"],
    )

    # Add computations as nodes and define edges
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = computations_dict[comp_name]

        # Ensure compliance with access and iterator constraints
        if len(comp_dict["accesses"]) > 15 or len(comp_dict["accesses"]) < 0:
            raise NbAccessException
        if len(comp_dict["iterators"]) > MAX_DEPTH:
            raise LoopsDepthException

        # Add node for computation
        node_id = f"C{comp_index}"
        graph["nodes"][node_id] = {
            "comp_name": comp_name,
            "is_reduction": comp_dict["comp_is_reduction"],
            "iterators": comp_dict["iterators"],
            "write_access": comp_dict["write_access_relation"],
            "read_accesses": comp_dict["accesses"],
        }

        # Add edges for read dependencies
        for read_access in comp_dict["accesses"]:
            buffer_id = read_access["buffer_id"]
            graph["edges"].append({
                "source": f"B{buffer_id}",
                "target": node_id,
                "type": "read"
            })

        # Add edge for write dependency
        write_buffer_id = comp_dict["write_buffer_id"]
        graph["edges"].append({
            "source": node_id,
            "target": f"B{write_buffer_id}",
            "type": "write"
        })

    # Add iterators as nodes
    iterators_dict = program_json["iterators"]
    for loop_name, loop_data in iterators_dict.items():
        node_id = f"L{loop_name}"
        graph["nodes"][node_id] = {
            "loop_name": loop_name,
            "lower_bound": loop_data["lower_bound"],
            "upper_bound": loop_data["upper_bound"],
        }

    return graph


def update_graph_attributes(graph, train_device="cpu"):
    for node_id, node_data in graph["nodes"].items():
        if "loop_name" in node_data:
            # Handle loop-specific attributes
            node_data["loop_index"] = torch.tensor(
                int(node_id[1:])
            ).to(train_device)
        if "comp_name" in node_data:
            # Handle computation-specific attributes
            node_data["comp_index"] = torch.tensor(
                int(node_id[1:])
            ).to(train_device)
            node_data["has_comps"] = bool(node_data.get("read_accesses") or node_data.get("write_access"))
    return graph


def get_schedule_representation(
    graph,
    schedule_json,
    comps_repr_templates_list,
    loops_repr_templates_list,
    comps_placeholders_indices_dict,
    loops_placeholders_indices_dict,
):
    # Create copies of templates to avoid modifying other schedules
    comps_repr = copy.deepcopy(comps_repr_templates_list)
    loops_repr = copy.deepcopy(loops_repr_templates_list)
    comps_expr_repr = []

    # Iterate through the computations in the graph
    for node_id, node_data in graph["nodes"].items():
        if "comp_name" not in node_data:
            continue  # Skip nodes that are not computations

        comp_name = node_data["comp_name"]
        comp_schedule_dict = schedule_json[comp_name]

        # Get the computation expression representation
        expr_dict = node_data["expression_representation"]
        comp_type = node_data["data_type"]
        expression_representation = get_graph_expr_repr(graph, node_id, comp_type)

        # Pad the expression representation
        expression_representation.extend(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * (MAX_EXPR_LEN - len(expression_representation))
        )
        comps_expr_repr.append(expression_representation)

        fused_levels = []
        if "fusions" in schedule_json and schedule_json["fusions"]:
            for fusion in schedule_json["fusions"]:
                if comp_name in fusion:
                    fused_levels.append(fusion[2])

        c_code = node_id
        # Loop representation for this computation
        for iter_i, iterator_name in enumerate(node_data["iterators"]):
            l_code = f"{c_code}-L{iter_i}"

            # Parallelization
            parallelized = 1 if iterator_name == comp_schedule_dict["parallelized_dim"] else 0
            p_index = comps_placeholders_indices_dict[l_code + "Parallelized"]
            comps_repr[p_index[0]][p_index[1]] = parallelized

            # Tiling
            tiled = 0
            tile_factor = 0
            if comp_schedule_dict["tiling"] and iterator_name in comp_schedule_dict["tiling"]["tiling_dims"]:
                tiled = 1
                tile_factor_index = comp_schedule_dict["tiling"]["tiling_dims"].index(iterator_name)
                tile_factor = int(comp_schedule_dict["tiling"]["tiling_factors"][tile_factor_index])
            p_index = comps_placeholders_indices_dict[l_code + "Tiled"]
            comps_repr[p_index[0]][p_index[1]] = tiled
            p_index = comps_placeholders_indices_dict[l_code + "TileFactor"]
            comps_repr[p_index[0]][p_index[1]] = tile_factor

            # Fusion
            fused = 1 if iter_i in fused_levels else 0
            p_index = comps_placeholders_indices_dict[l_code + "Fused"]
            comps_repr[p_index[0]][p_index[1]] = fused

            # Shifting
            shifted = 0
            shifting_factor = 0
            if comp_schedule_dict["shiftings"]:
                for shifting in comp_schedule_dict["shiftings"]:
                    if iterator_name.startswith(shifting[0]):
                        shifted = 1
                        shifting_factor = shifting[1]
                        break
            p_index = comps_placeholders_indices_dict[l_code + "Shifted"]
            comps_repr[p_index[0]][p_index[1]] = shifted
            p_index = comps_placeholders_indices_dict[l_code + "ShiftFactor"]
            comps_repr[p_index[0]][p_index[1]] = shifting_factor

        # Unrolling
        unrolled = 1 if comp_schedule_dict["unrolling_factor"] else 0
        unroll_factor = int(comp_schedule_dict["unrolling_factor"] or 0)
        p_index = comps_placeholders_indices_dict[c_code + "-Unrolled"]
        comps_repr[p_index[0]][p_index[1]] = unrolled
        p_index = comps_placeholders_indices_dict[c_code + "-UnrollFactor"]
        comps_repr[p_index[0]][p_index[1]] = unroll_factor

        # Transformations
        padded_tags = get_padded_transformation_tags(graph, schedule_json, comp_name)
        tags_start = comps_placeholders_indices_dict[c_code + "-TransformationTagsStart"]
        tags_end = comps_placeholders_indices_dict[c_code + "-TransformationTagsEnd"]
        nb_tags_elements = tags_end[1] - tags_start[1] + 1
        assert len(padded_tags) == nb_tags_elements
        comps_repr[tags_start[0]][tags_start[1]:tags_end[1] + 1] = padded_tags

    # Fill loop representations
    for loop_name, loop_data in graph["nodes"].items():
        if "loop_name" not in loop_data:
            continue  # Skip nodes that are not loops
        l_code = f"L{loop_name}"
        loop_schedule_dict = schedule_json.get(loop_name, {})
        p_index = loops_placeholders_indices_dict[l_code + "Parallelized"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedule_dict.get("parallelized", 0)

        p_index = loops_placeholders_indices_dict[l_code + "Tiled"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedule_dict.get("tiled", 0)
        p_index = loops_placeholders_indices_dict[l_code + "TileFactor"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedule_dict.get("tile_factor", 0)

    computations_tensor = torch.unsqueeze(torch.FloatTensor(comps_repr), 0)
    loops_tensor = torch.unsqueeze(torch.FloatTensor(loops_repr), 0)
    comps_expr_repr = torch.tensor([comps_expr_repr]).float()

    return computations_tensor, loops_tensor, comps_expr_repr


def get_func_repr_task(input_q, output_q):
    process_id, programs_dict, repr_pkl_output_folder, train_device = input_q.get()
    function_name_list = list(programs_dict.keys())
    local_list = []

    for function_name in function_name_list:
        program = programs_dict[function_name]

        # Construct the graph representation
        graph = get_representation_template(program, train_device)
        graph = update_graph_attributes(graph, train_device)

        program_exec_time = program["initial_execution_time"]

        # Prepare data for each schedule
        for schedule_index, schedule_json in enumerate(program["schedules_list"]):
            sched_exec_time = min(schedule_json["execution_times"])
            if drop_schedule(program, schedule_index) or not sched_exec_time:
                continue

            sched_speedup = speedup_clip(program_exec_time / sched_exec_time)

            comps_tensor, loops_tensor, comps_expr_repr = get_schedule_representation(
                graph,
                schedule_json,
                program["comps_repr_templates_list"],
                program["loops_repr_templates_list"],
                program["comps_placeholders_indices_dict"],
                program["loops_placeholders_indices_dict"],
            )

            local_list.append({
                "function_name": function_name,
                "comps_tensor": comps_tensor,
                "loops_tensor": loops_tensor,
                "comps_expr_tree": comps_expr_repr,
                "sched_speedup": sched_speedup,
            })

    pkl_part_filename = f"{repr_pkl_output_folder}/pickled_representation_part_{process_id}.pkl"
    with open(pkl_part_filename, "wb") as f:
        pickle.dump(local_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    output_q.put((process_id, pkl_part_filename))


class Dataset_parallel:
    def __init__(
        self,
        dataset_filename,
        max_batch_size=1024,
        drop_sched_func=None,
        drop_prog_func=None,
        speedups_clip_func=None,
        no_batching=False,
        store_device="cpu",
        train_device="cpu",
        repr_pkl_output_folder="none",
        just_load_pickled_repr=False,
        nb_processes=15,
        min_functions_per_graph_footprint=0,
    ):
        # Structures to hold data
        self.batched_X = []
        self.batched_Y = []
        self.nb_dropped = 0
        self.nb_pruned = 0
        self.dropped_funcs = []
        self.batched_datapoint_attributes = []
        self.nb_datapoints = 0
        self.gpu_fitted_batches_index = -1
        self.nb_funcs_per_footprint = {}

        processes_output_list = []
        programs_dict = {}
        batches_dict = dict()

        if just_load_pickled_repr:
            # Load preprocessed graph representations
            for pkl_part_filename in tqdm(list(Path(repr_pkl_output_folder).iterdir())):
                with open(str(pkl_part_filename), "rb") as f:
                    lst = pickle.load(f)
                processes_output_list.extend(lst)
        else:
            # Split functions for parallel processing
            manager = multiprocessing.Manager()
            input_queue = manager.Queue()
            output_queue = manager.Queue()

            processes = [
                multiprocessing.Process(
                    target=get_func_repr_task, args=(input_queue, output_queue)
                )
                for _ in range(nb_processes)
            ]

            for process in processes:
                process.start()

            # Load dataset
            if dataset_filename.endswith("json"):
                with open(dataset_filename, "r") as f:
                    dataset_str = f.read()
                programs_dict = json.loads(dataset_str)
            elif dataset_filename.endswith("pkl"):
                with open(dataset_filename, "rb") as f:
                    programs_dict = pickle.load(f)

            functions_list = list(programs_dict.keys())
            random.Random(42).shuffle(functions_list)

            nb_funcs_per_process = (len(functions_list) // nb_processes) + 1

            for i in range(nb_processes):
                process_programs_dict = dict(
                    list(programs_dict.items())[
                        i * nb_funcs_per_process : (i + 1) * nb_funcs_per_process
                    ]
                )
                input_queue.put(
                    (i, process_programs_dict, repr_pkl_output_folder, store_device)
                )

            for _ in range(nb_processes):
                process_id, pkl_part_filename = output_queue.get()
                if not no_batching:
                    with open(pkl_part_filename, "rb") as f:
                        lst = pickle.load(f)
                    processes_output_list.extend(lst)

        # Define default behavior for optional functions
        drop_sched_func = drop_sched_func or (lambda x, y: False)
        drop_prog_func = drop_prog_func or (lambda x, y: False)
        speedups_clip_func = speedups_clip_func or (lambda x: x)

        if no_batching:
            print("Parameter no_batching is True. Stopping after the PKL files were saved.")
            return

        print("Assembling schedules from each function")
        for (
            function_name,
            nb_dropped,
            graph_footprint,
            local_function_dict,
        ) in processes_output_list:
            batches_dict[graph_footprint] = batches_dict.get(
                graph_footprint,
                {
                    "graph": local_function_dict["graph"],
                    "comps_tensor_list": [],
                    "loops_tensor_list": [],
                    "datapoint_attributes_list": [],
                    "comps_expr_tree_list": [],
                    "speedups_list": [],
                },
            )
            batches_dict[graph_footprint]["comps_tensor_list"].extend(
                local_function_dict["comps_tensor_list"]
            )
            batches_dict[graph_footprint]["loops_tensor_list"].extend(
                local_function_dict["loops_tensor_list"]
            )
            batches_dict[graph_footprint]["datapoint_attributes_list"].extend(
                local_function_dict["datapoint_attributes_list"]
            )
            batches_dict[graph_footprint]["comps_expr_tree_list"].extend(
                local_function_dict["comps_expr_tree_list"]
            )
            batches_dict[graph_footprint]["speedups_list"].extend(
                local_function_dict["speedups_list"]
            )

            self.nb_dropped += nb_dropped
            self.nb_datapoints += len(local_function_dict["speedups_list"])
            self.nb_funcs_per_footprint[graph_footprint] = self.nb_funcs_per_footprint.get(
                graph_footprint, {"nb_funcs": 0, "nb_dps": 0}
            )
            self.nb_funcs_per_footprint[graph_footprint]["nb_funcs"] += 1
            self.nb_funcs_per_footprint[graph_footprint]["nb_dps"] += len(
                local_function_dict["speedups_list"]
            )

        del processes_output_list, programs_dict
        gc.collect()

        print("Batching data")
        storing_device = torch.device(store_device)
        for graph_footprint in tqdm(batches_dict):
            if (
                self.nb_funcs_per_footprint[graph_footprint]["nb_funcs"]
                < min_functions_per_graph_footprint
                and self.nb_funcs_per_footprint[graph_footprint]["nb_dps"]
                < 100 * min_functions_per_graph_footprint
            ):
                self.nb_datapoints -= self.nb_funcs_per_footprint[graph_footprint]["nb_dps"]
                continue

            zipped = list(
                zip(
                    batches_dict[graph_footprint]["datapoint_attributes_list"],
                    batches_dict[graph_footprint]["comps_tensor_list"],
                    batches_dict[graph_footprint]["comps_expr_tree_list"],
                    batches_dict[graph_footprint]["loops_tensor_list"],
                    batches_dict[graph_footprint]["speedups_list"],
                )
            )
            random.shuffle(zipped)
            (
                batches_dict[graph_footprint]["datapoint_attributes_list"],
                batches_dict[graph_footprint]["comps_tensor_list"],
                batches_dict[graph_footprint]["comps_expr_tree_list"],
                batches_dict[graph_footprint]["loops_tensor_list"],
                batches_dict[graph_footprint]["speedups_list"],
            ) = zip(*zipped)

            for chunk in range(
                0,
                len(batches_dict[graph_footprint]["speedups_list"]),
                max_batch_size,
            ):
                if (
                    storing_device.type == "cuda"
                    and (
                        torch.cuda.memory_allocated(storing_device.index)
                        / torch.cuda.get_device_properties(storing_device.index).total_memory
                    )
                    > 0.83
                ):
                    print(
                        f"GPU memory on {storing_device} nearly full, switching to CPU memory"
                    )
                    self.gpu_fitted_batches_index = len(self.batched_X)
                    storing_device = torch.device("cpu")

                self.batched_datapoint_attributes.append(
                    batches_dict[graph_footprint]["datapoint_attributes_list"][
                        chunk : chunk + max_batch_size
                    ]
                )
                x = torch.cat(
                    batches_dict[graph_footprint]["comps_tensor_list"][
                        chunk : chunk + max_batch_size
                    ],
                    0,
                )
                batch_size, num_comps, __dict__ = x.shape
                x = x.view(batch_size * num_comps, -1)
                first_part, vectors, third_part = separate_vector(
                    x, num_transformations=4, pad=False
                )
                self.batched_X.append(
                    (
                        batches_dict[graph_footprint]["graph"],
                        first_part.to(storing_device).view(batch_size, num_comps, -1),
                        vectors.to(storing_device),
                        third_part.to(storing_device).view(batch_size, num_comps, -1),
                        torch.cat(
                            batches_dict[graph_footprint]["loops_tensor_list"][
                                chunk : chunk + max_batch_size
                            ],
                            0,
                        ).to(storing_device),
                        torch.cat(
                            batches_dict[graph_footprint]["comps_expr_tree_list"][
                                chunk : chunk + max_batch_size
                            ],
                            0,
                        ).to(storing_device),
                    )
                )
                self.batched_Y.append(
                    torch.FloatTensor(
                        batches_dict[graph_footprint]["speedups_list"][
                            chunk : chunk + max_batch_size
                        ]
                    ).to(storing_device)
                )

            del (
                batches_dict[graph_footprint]["comps_tensor_list"],
                batches_dict[graph_footprint]["loops_tensor_list"],
                batches_dict[graph_footprint]["comps_expr_tree_list"],
                batches_dict[graph_footprint]["speedups_list"],
                batches_dict[graph_footprint]["datapoint_attributes_list"],
            )

        del batches_dict
        gc.collect()

        if self.gpu_fitted_batches_index == -1:
            self.gpu_fitted_batches_index = len(self.batched_X)

        print(
            f"Number of datapoints: {self.nb_datapoints}, Number of batches: {len(self.batched_Y)}, Dropped points: {self.nb_dropped}"
        )

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(index, int):
            return self.batched_X[index], self.batched_Y[index]

    def __len__(self):
        return len(self.batched_Y)
   

def load_pickled_repr(
    repr_pkl_output_folder=None,
    max_batch_size=1024,
    store_device="cpu",
    train_device="cpu",
    min_functions_per_graph_footprint=0,
):
    dataset = Dataset_parallel(
        None,
        max_batch_size,
        None,
        repr_pkl_output_folder=repr_pkl_output_folder,
        just_load_pickled_repr=True,
        store_device=store_device,
        train_device=train_device,
        min_functions_per_tree_footprint=min_functions_per_graph_footprint,
    )

    indices = list(range(len(dataset)))
    batches_list = [dataset[i] for i in indices]

    return dataset, batches_list, indices, dataset.gpu_fitted_batches_index

def load_data_into_pkls_parallel(
    train_val_dataset_file,
    nb_processes=15,
    repr_pkl_output_folder=None,
    overwrite_existing_pkl=False,
):
    if Path(repr_pkl_output_folder).is_dir() and overwrite_existing_pkl:
        shutil.rmtree(repr_pkl_output_folder)
        print("Deleted existing folder", repr_pkl_output_folder)

    Path(repr_pkl_output_folder).mkdir(parents=True, exist_ok=False)
    print("Created folder", repr_pkl_output_folder)

    print("Loading data from:", train_val_dataset_file)
    dataset = Dataset_parallel(
        train_val_dataset_file,
        no_batching=True,
        just_load_pickled_repr=False,
        nb_processes=nb_processes,
        repr_pkl_output_folder=repr_pkl_output_folder,
        store_device="cpu",
        train_device="cpu",
    )
    return


def get_graph_footprint(graph):
    footprint = ""
    for node_id, node_data in graph["nodes"].items():
        if "comp_name" in node_data:
            footprint += f"<C{node_id}>"
        elif "loop_name" in node_data:
            footprint += f"<L{node_id}>"
    for edge in graph["edges"]:
        footprint += f"({edge['source']}->{edge['target']})"
    return footprint


def get_padded_transformation_tags(graph, schedule_json, comp_name):
    node = [n for n, d in graph["nodes"].items() if d.get("comp_name") == comp_name][0]
    transformations = schedule_json[comp_name]["transformations_list"]
    tags_list = []

    for transformation in transformations:
        tags_list.extend(transformation)

    # Pad to match the maximum number of transformations
    tags_list += [0] * (MAX_NUM_TRANSFORMATIONS * MAX_TAGS - len(tags_list))
    return tags_list


def get_datapoint_attributes(func_name, program_dict, schedule_index, graph_footprint):
    schedule_json = program_dict["schedules_list"][schedule_index]
    sched_id = str(schedule_index).zfill(4)
    sched_str = get_schedule_str(program_dict["program_annotation"], schedule_json)
    exec_time = np.min(schedule_json["execution_times"])
    memory_use = program_dict["program_annotation"]["memory_size"]
    node_name = program_dict.get("node_name", "unknown")
    speedup = program_dict["initial_execution_time"] / exec_time

    return (
        func_name,
        sched_id,
        sched_str,
        exec_time,
        memory_use,
        node_name,
        graph_footprint,
        speedup,
    )


def pad_access_matrix(access_matrix):
    access_matrix = np.array(access_matrix)
    access_matrix = np.c_[np.ones(access_matrix.shape[0]), access_matrix]
    access_matrix = np.r_[[np.ones(access_matrix.shape[1])], access_matrix]
    padded_access_matrix = np.zeros((MAX_DEPTH + 1, MAX_DEPTH + 2))
    padded_access_matrix[: access_matrix.shape[0], : access_matrix.shape[1] - 1] = access_matrix[:, :-1]
    padded_access_matrix[: access_matrix.shape[0], -1] = access_matrix[:, -1]
    return padded_access_matrix


def isl_to_write_matrix(isl_map):
    comp_iterators_str = re.findall(r"\[(.*)\]\s*->", isl_map)[0]
    buffer_iterators_str = re.findall(r"->\s*\w*\[(.*)\]", isl_map)[0]
    buffer_iterators_str = re.sub(r"\w+'\s=", "", buffer_iterators_str)
    comp_iter_names = re.findall(r"(?:\s*(\w+))+", comp_iterators_str)
    buf_iter_names = re.findall(r"(?:\s*(\w+))+", buffer_iterators_str)
    matrix = np.zeros([len(buf_iter_names), len(comp_iter_names) + 1])
    for i, buf_iter in enumerate(buf_iter_names):
        for j, comp_iter in enumerate(comp_iter_names):
            if buf_iter == comp_iter:
                matrix[i, j] = 1
                break
    return matrix


def isl_to_write_dims(isl_map):
    buffer_iterators_str = re.findall(r"->\s*\w*\[(.*)\]", isl_map)[0]
    buffer_iterators_str = re.sub(r"\w+'\s=", "", buffer_iterators_str)
    buf_iter_names = re.findall(r"(?:\s*(\w+))+", buffer_iterators_str)
    return buf_iter_names

def get_results_df(
    dataset, batches_list, indices, model, log=False, train_device="cpu"
):
    df = pd.DataFrame()
    model.eval()
    torch.set_grad_enabled(False)
    all_outputs = []
    all_labels = []
    prog_names = []
    sched_names = []
    exec_times = []
    sched_strs = []
    memory_uses = []
    node_names = []
    tree_footprints = []

    for k, (inputs, labels) in tqdm(list(enumerate(batches_list))):
        original_device = labels.device
        inputs = (
                    inputs[0],
                    inputs[1].to(train_device),
                    inputs[2].to(train_device),
                    inputs[3].to(train_device),
                    inputs[4].to(train_device),
                    inputs[5].to(train_device),
                )
        labels = labels.to(train_device)
        outputs = model(inputs)
        assert outputs.shape == labels.shape
        all_outputs.append(outputs)
        all_labels.append(labels)

        assert len(outputs) == len(dataset.batched_datapoint_attributes[indices[k]])
        zipped_attributes = list(zip(*dataset.batched_datapoint_attributes[indices[k]]))
        prog_names.extend(zipped_attributes[0])
        sched_names.extend(zipped_attributes[1])
        sched_strs.extend(zipped_attributes[2])
        exec_times.extend(zipped_attributes[3])
        memory_uses.extend(zipped_attributes[4])
        node_names.extend(zipped_attributes[5])
        tree_footprints.extend(zipped_attributes[6])
        inputs = (
                    inputs[0],
                    inputs[1].to(original_device),
                    inputs[2].to(original_device),
                    inputs[3].to(original_device),
                    inputs[4].to(original_device),
                    inputs[5].to(original_device),
                )
        labels = labels.to(original_device)
    preds = torch.cat(all_outputs)
    targets = torch.cat(all_labels)
    preds = preds.cpu().detach().numpy().reshape((-1,))
    preds = np.around(preds, decimals=6)
    targets = np.around(targets.cpu().detach().numpy().reshape((-1,)), decimals=6)

    assert preds.shape == targets.shape
    df["name"] = prog_names
    df["tree_struct"] = tree_footprints
    df["sched_name"] = sched_names
    df["sched_str"] = sched_strs
    df["exec_time"] = exec_times
    df["memory_use"] = list(map(float, memory_uses))
    df["node_name"] = node_names
    df["prediction"] = np.array(preds)
    df["target"] = np.array(targets)

    df["APE"] = np.abs(df.target - df.prediction) / df.target * 100
    return df

# Calculate the Normalized Discounted Cumulative Gain while only considiring the top rated schedule (k=1)
def function_wise_ndcg_1(g):
    if len(g["target"]) < 2:
        return pd.Series(dict(nDCG_1=np.nan))
    score = ndcg_score([g["target"].tolist()], [g["prediction"].tolist()], k=1)
    return pd.Series(dict(nDCG_1=score))

# Calculate the Normalized Discounted Cumulative Gain while only considiring the top 5 rated schedules (k=5)
def function_wise_ndcg_5(g):
    if len(g["target"]) < 2:
        return pd.Series(dict(nDCG_5=np.nan))
    score = ndcg_score([g["target"].tolist()], [g["prediction"].tolist()], k=5)
    return pd.Series(dict(nDCG_5=score))

# Calculate the Normalized Discounted Cumulative Gain while considiring all the schedules
def function_wise_ndcg_full(g):
    if len(g["target"]) < 2:
        return pd.Series(dict(nDCG=np.nan))
    score = ndcg_score([g["target"].tolist()], [g["prediction"].tolist()], k=None)
    return pd.Series(dict(nDCG=score))

# Calculate the Spearman correlation coefficient
def function_wise_spearman(g):
    if len(g["target"]) < 2:
        return pd.Series(dict(Spearman_r=np.nan))
    score = spearmanr(g["target"], g["prediction"])[0]
    return pd.Series(dict(Spearman_r=score))

# Calculate the absolute percentage error
def function_wise_ape(g):
    score = np.mean(g["APE"])
    return pd.Series(dict(MAPE=score))

# calculates the model scores from the dataframe
def get_scores(df):
    with tqdm(total=6) as pbar:
        df_spearman = df.groupby("name").apply(function_wise_spearman).reset_index()
        pbar.update(1)
        df_mape = df.groupby("name").apply(function_wise_ape).reset_index()
        pbar.update(1)
        df_ndcg = df.groupby("name").apply(function_wise_ndcg_full).reset_index()
        pbar.update(1)
        df_ndcg1 = df.groupby("name").apply(function_wise_ndcg_1).reset_index()
        pbar.update(1)
        df_ndcg5 = df.groupby("name").apply(function_wise_ndcg_5).reset_index()
        pbar.update(1)
        df_count = df.groupby("name").agg("count").reset_index()[["name", "sched_name"]]
        df_count.columns = ["name", "count"]
        pbar.update(1)

    scores_df = (
        df_count.merge(df_ndcg, on="name")
        .merge(df_ndcg5, on="name")
        .merge(df_ndcg1, on="name")
        .merge(df_spearman, on="name")
        .merge(df_mape, on="name")
    )
    return scores_df

# Solving the Linear Diophantine equation & finding basic solution (sigma & gamma) for : f_i* sigma - f_j*gamma = 1
# Used to get skewing parameters
def linear_diophantine_default(f_i, f_j):
    n1 = abs(f_i)
    n2 = abs(f_j)
    
    while(n1 != n2):
        if(n1 > n2):
            n1 -=  n2
        else:
            n2 -=  n1
            
    # Update f_i and f_j to equivalent but prime between themselfs value
    f_i = f_i / n1
    f_j = f_j / n1
    
    found = False
    gamma = 0
    sigma = 1
    
    if (f_j == 1) or (f_i == 1):
        gamma = f_i - 1
        sigma = 1
        # Since sigma = 1  then
        # f_i - gamma * f_j = 1 & using the previous condition :
        #  - f_i = 1 : then gamma = 0 (f_i-1) is enough
        #  - f_j = 1 : then gamma = f_i -1  
    else:
        if (f_j == -1) and (f_i > 1):
            gamma = 1
            sigma = 0
        else:
            # General case : solving the Linear Diophantine equation & finding basic solution (sigma & gamma) for : f_i* sigma - f_j*gamma = 1
            i = 0
            while (i < 100) and (not found):
                if ((sigma * f_i) % abs(f_j)) == 1:
                    found = True
                else:
                    sigma += 1
                    i += 1
            if not found:
                # Detect infinite loop and prevent it in case where f_i and f_j are not prime between themselfs
                print("Error cannof find solution to diophantine equation")
                return
            gamma = ((sigma * f_i) - 1) / f_j
    return gamma, sigma

# Set a lower bound for speedups to avoid fluctuations and make the training easier
def speedup_clip(speedup):
    if speedup < 0.01:
        speedup = 0.01
    return speedup

# Check if this program should be dropped
def drop_program(prog_dict, prog_name):
    # If there are no schedules explored for this function
    if len(prog_dict["schedules_list"]) < 2:
        return True
    
    return False

# Check if this schedule should be dropped
def drop_schedule(prog_dict, schedule_index):
    schedule_json = prog_dict["schedules_list"][schedule_index]
    # If the execution list is empty or it contains incoherent executions 
    if (not schedule_json["execution_times"]) or min(schedule_json["execution_times"]) < 0: 
        return True
    
    return False

def get_involved_comps(node, graph):
    """
    Retrieve the computations involved in the current node.
    """
    result = []
    if not node or node not in graph["nodes"]:
        return result

    node_data = graph["nodes"][node]
    if "comp_name" in node_data:
        result.append(node_data["comp_name"])

    # Traverse edges to find children
    for edge in graph["edges"]:
        if edge["source"] == node and edge["type"] == "hierarchical":
            result.extend(get_involved_comps(edge["target"], graph))
    return result


def get_comp_iterators_from_graph(graph, comp_name):
    """
    Retrieve the iterators associated with a specific computation in a DAG.
    """
    iterators = []
    to_explore = []

    # Identify the root nodes that contain the computation
    for node_id, node_data in graph["nodes"].items():
        if "comp_name" in node_data and comp_name in get_involved_comps(node_id, graph):
            to_explore.append(node_id)

    while to_explore:
        current_node = to_explore.pop(0)
        current_data = graph["nodes"][current_node]

        if "loop_name" in current_data:
            iterators.append(current_data["loop_name"])

        # Add child nodes connected hierarchically
        for edge in graph["edges"]:
            if edge["source"] == current_node and edge["type"] == "hierarchical":
                to_explore.append(edge["target"])

    return iterators

def get_expr_repr(expr, comp_type):
    """
    Generate one-hot encoding for an expression and its data type.
    """
    expr_mapping = {
        "add": [1, 0, 0, 0, 0, 0, 0, 0],
        "sub": [0, 1, 0, 0, 0, 0, 0, 0],
        "mul": [0, 0, 1, 0, 0, 0, 0, 0],
        "div": [0, 0, 0, 1, 0, 0, 0, 0],
        "sqrt": [0, 0, 0, 0, 1, 0, 0, 0],
        "min": [0, 0, 0, 0, 0, 1, 0, 0],
        "max": [0, 0, 0, 0, 0, 0, 1, 0],
    }
    expr_vector = expr_mapping.get(expr, [0, 0, 0, 0, 0, 0, 0, 1])

    comp_type_mapping = {
        "int32": [1, 0, 0],
        "float32": [0, 1, 0],
        "float64": [0, 0, 1],
    }
    comp_type_vector = comp_type_mapping.get(comp_type, [0, 0, 0])

    return expr_vector + comp_type_vector


def get_graph_expr_repr(node_id, graph, comp_type):
    """
    Generate the representation of an expression in a DAG structure.
    """
    expr_tensor = []
    node_data = graph["nodes"].get(node_id, {})

    # Recursively process child nodes
    for edge in graph["edges"]:
        if edge["source"] == node_id and edge["type"] == "hierarchical":
            expr_tensor.extend(get_graph_expr_repr(edge["target"], graph, comp_type))

    # Add the current node's expression representation
    if "expr_type" in node_data:
        expr_tensor.append(get_expr_repr(node_data["expr_type"], comp_type))

    return expr_tensor


def get_padded_initial_iteration_domain(graph, comp_name, pad=True):
    comp_node = next(
        (n for n, data in graph["nodes"].items() if data.get("comp_name") == comp_name),
        None,
    )
    if not comp_node:
        raise ValueError(f"Computation {comp_name} not found in the graph.")

    comp_data = graph["nodes"][comp_node]
    iterators = comp_data["iterators"]
    nb_dims = len(iterators)

    coeff_mat = np.zeros((nb_dims * 2, nb_dims), int)
    constants_col = np.zeros((nb_dims * 2), int)

    for i, iterator_name in enumerate(iterators):
        coeff_mat[i * 2, i] = -1
        coeff_mat[i * 2 + 1, i] = 1

        iterator_data = graph["nodes"].get(iterator_name, {})
        upper_bound = str(sympy.simplify(iterator_data["upper_bound"])).replace(" ", "")
        lower_bound = str(sympy.simplify(iterator_data["lower_bound"])).replace(" ", "")

        if "max" in lower_bound or "Max" in lower_bound:
            lower_bound = re.findall(r"[mM]ax\(.+,(.+)\)", lower_bound)[0]

        iterators_in_upper = re.findall(r"[a-zA-Z]\w*", upper_bound)
        constants_in_upper = re.findall(r"(?:^|\+|-)\d+", upper_bound) or ["0"]
        iterators_in_lower = re.findall(r"[a-zA-Z]\w*", lower_bound)
        constants_in_lower = re.findall(r"(?:^|\+|-)\d+", lower_bound) or ["0"]

        for iter_name in iterators_in_upper:
            col_idx = iterators.index(iter_name)
            coeff_mat[i * 2 + 1, col_idx] = 1 if f"-{iter_name}" in upper_bound else -1
        constants_col[i * 2 + 1] = int(constants_in_upper[0]) - 1

        for iter_name in iterators_in_lower:
            col_idx = iterators.index(iter_name)
            coeff_mat[i * 2, col_idx] = -1 if f"-{iter_name}" in lower_bound else 1
        constants_col[i * 2] = -int(constants_in_lower[0])

    if pad:
        padded_coeff_mat = np.pad(
            coeff_mat,
            [(0, MAX_DEPTH * 2 - nb_dims * 2), (0, MAX_DEPTH - nb_dims)],
            mode="constant",
        )
        padded_constants_col = np.pad(
            constants_col, [(0, MAX_DEPTH * 2 - nb_dims * 2)], mode="constant"
        )
        return padded_coeff_mat, padded_constants_col
    else:
        return coeff_mat, constants_col

    
def get_padded_transformed_iteration_domain(graph, schedule_json, comp_name, pad=True):
    transformation_matrix = get_transformation_matrix(graph, schedule_json, comp_name)

    A, b = get_padded_initial_iteration_domain(graph, comp_name, pad=False)
    inverse = np.linalg.inv(transformation_matrix)
    result = np.matmul(A, inverse)

    if pad:
        result = np.pad(
            result,
            [(0, MAX_DEPTH * 2 - result.shape[0]), (0, MAX_DEPTH - result.shape[1])],
            mode="constant",
        )
    return result


def get_transformation_matrix(graph, schedule_json, comp_name):
    comp_node = next(
        (n for n, data in graph["nodes"].items() if data.get("comp_name") == comp_name),
        None,
    )
    if not comp_node:
        raise ValueError(f"Computation {comp_name} not found in the graph.")

    iterators = graph["nodes"][comp_node]["iterators"]
    nb_iterators = len(iterators)
    final_transformation = np.identity(nb_iterators)

    for transformation in schedule_json[comp_name]["transformations_list"]:
        matrix = get_transformation_matrix_from_vector(transformation, nb_iterators)
        final_transformation = np.matmul(matrix, final_transformation)
    return final_transformation

def get_transformation_matrix_from_vector(transformation, matrix_size):
    matrix = np.identity(matrix_size)
    assert len(transformation) == MAX_TAGS

    if transformation[0] == 1:
        # Interchange
        assert transformation[1] < matrix_size and transformation[2] < matrix_size
        matrix[transformation[1], transformation[2]] = 1
        matrix[transformation[1], transformation[1]] = 0
        matrix[transformation[2], transformation[1]] = 1
        matrix[transformation[2], transformation[2]] = 0

    elif transformation[0] == 2:
        # Reversal
        assert transformation[3] < matrix_size
        matrix[transformation[3], transformation[3]] = -1

    elif transformation[0] == 3:
        # Skewing
        dim_count = sum(1 for d in transformation[4:7] if d != 0)
        assert dim_count <= matrix_size
        skew_params = transformation[7 : 7 + dim_count ** 2]
        skew_matrix = np.array(skew_params).reshape((dim_count, dim_count))
        for i in range(dim_count):
            for j in range(dim_count):
                matrix[transformation[4 + i], transformation[4 + j]] = skew_matrix[i, j]

    return matrix


def get_schedule_str(program_json, sched_json, graph: nx.DiGraph):
    comp_name = [
        n
        for n in sched_json.keys()
        if not n in ["unfuse_iterators", "tree_structure", "execution_times", "fusions", "sched_str", "legality_check", "exploration_method"]
    ]
    sched_str = ""
    
    if ("fusions" in sched_json and sched_json["fusions"]):
        for fusion in sched_json["fusions"]:
            sched_str += "F("
            for name in comp_name:
                if name in fusion:
                    sched_str += name + ","
            sched_str = sched_str[:-1]
            sched_str += ")"

    for name in comp_name:
        transf_loop_nest = program_json["computations"][name]["iterators"].copy()
        schedule = sched_json[name]
        
        # Handle fusions using graph edges (dependencies)
        if "fusions" in sched_json and sched_json["fusions"]:
            for fusion in sched_json["fusions"]:
                if name in fusion:
                    iterator_comp_name = fusion[0]
                    transf_loop_nest = program_json["computations"][iterator_comp_name]["iterators"].copy()
                    schedule = sched_json[iterator_comp_name]

        sched_str += '{' + name + '}:' 

        for transformation in schedule["transformations_list"]:
            if (transformation[0] == 1):
                sched_str += "I(L" + str(transformation[1]) + ",L" + str(transformation[2]) + ")"
                assert(transformation[1] < len(transf_loop_nest) and transformation[2] < len(transf_loop_nest))
                tmp_it = transf_loop_nest[transformation[1]]
                transf_loop_nest[transformation[1]] = transf_loop_nest[transformation[2]]
                transf_loop_nest[transformation[2]] = tmp_it
                
            elif (transformation[0] == 2):
                sched_str += "R(L" + str(transformation[3])+ ")"
            elif (transformation[0] == 3):
                sched_str += "S(L" + str(transformation[4]) + ",L" + str(transformation[5]) + "," + str(transformation[6]) + "," + str(transformation[7]) + ")"

        # Use graph traversal to handle dependencies and parallelization
        if schedule["parallelized_dim"]:
            dim_index = transf_loop_nest.index(schedule["parallelized_dim"])
            sched_str += "P(L" + str(dim_index) + ")"
            
        if schedule["shiftings"]:    
            for shifting in schedule['shiftings']: 
                dim_index = transf_loop_nest.index(shifting[0])
                sched_str += "Sh(L" + str(dim_index) + "," + str(shifting[1]) + ")"
                
        if schedule["tiling"]:
            if schedule["tiling"]["tiling_depth"] == 1:
                first_dim = schedule["tiling"]["tiling_dims"][0]
                first_dim_index = transf_loop_nest.index(first_dim)
                first_factor = schedule["tiling"]["tiling_factors"][0]
                sched_str += "T1(L" + str(first_dim_index) + "," + str(first_factor) + ")"
                transf_loop_nest[first_dim_index] = first_dim + "_outer", first_dim + "_inner"
            elif schedule["tiling"]["tiling_depth"] == 2:
                first_dim = schedule["tiling"]["tiling_dims"][0]
                second_dim = schedule["tiling"]["tiling_dims"][1]
                first_dim_index = transf_loop_nest.index(first_dim)
                second_dim_index = transf_loop_nest.index(second_dim)
                first_factor = schedule["tiling"]["tiling_factors"][0]
                second_factor = schedule["tiling"]["tiling_factors"][1]
                sched_str += "T2(L" + str(first_dim_index) + ",L" + str(second_dim_index) + "," + str(first_factor) + "," + str(second_factor) + ")"
                transf_loop_nest[first_dim_index] = first_dim + "_outer", second_dim + "_outer"
                transf_loop_nest[second_dim_index] = first_dim + "_inner", second_dim + "_inner"
            elif schedule["tiling"]["tiling_depth"] == 3:
                first_dim = schedule["tiling"]["tiling_dims"][0]
                second_dim = schedule["tiling"]["tiling_dims"][1]
                third_dim = schedule["tiling"]["tiling_dims"][2]
                first_dim_index = transf_loop_nest.index(first_dim)
                second_dim_index = transf_loop_nest.index(second_dim)
                third_dim_index = transf_loop_nest.index(third_dim)
                first_factor = schedule["tiling"]["tiling_factors"][0]
                second_factor = schedule["tiling"]["tiling_factors"][1]
                third_factor = schedule["tiling"]["tiling_factors"][2]
                sched_str += "T3(L" + str(first_dim_index) + ",L" + str(second_dim_index) + ",L" + str(third_dim_index) + "," + str(first_factor) + "," + str(second_factor) + "," + str(third_factor) + ")"
                transf_loop_nest[first_dim_index] = first_dim + "_outer", second_dim + "_outer", third_dim + "_outer"
                transf_loop_nest[second_dim_index] = first_dim + "_inner", second_dim + "_inner", third_dim + "_inner"
                transf_loop_nest.remove(third_dim)

        if schedule["unrolling_factor"]:
            dim_index = len(transf_loop_nest) - 1
            dim_name = transf_loop_nest[-1]
            sched_str += "U(L" + str(dim_index) + "," + schedule["unrolling_factor"] + ")"
            transf_loop_nest[dim_index : dim_index + 1] = (dim_name + "_Uouter", dim_name + "_Uinner")

    return sched_str



def seperate_vector(X: torch.Tensor, num_transformations: int = 4, pad: bool = True, pad_amount: int = 5, graph: nx.Graph = None) -> torch.Tensor:
    batch_size, _ = X.shape
    first_part = X[:, :33]
    second_part = X[:, 33 : 33 + MAX_TAGS * num_transformations]
    third_part = X[:, 33 + MAX_TAGS * num_transformations :]

    vectors = []
    for i in range(num_transformations):
        vector = second_part[:, MAX_TAGS * i : MAX_TAGS * (i + 1)].reshape(batch_size, 1, -1)
        vectors.append(vector)

    if pad:
        for i in range(pad_amount):
            vector = torch.zeros_like(vector)
            vectors.append(vector)

    return (first_part, torch.cat(vectors[0:], dim=1), third_part)


def graph_indices_to_device(node, train_device, graph: nx.DiGraph):
    node['loop_index'] = node['loop_index'].to(train_device, non_blocking=True)
    if 'computations_indices' in node:
        node['computations_indices'] = node['computations_indices'].to(train_device, non_blocking=True)
    
    # Traverse graph using BFS or DFS instead of assuming a tree structure
    for child in graph.neighbors(node['id']):
        child_node = graph.nodes[child]
        graph_indices_to_device(child_node, train_device, graph)
