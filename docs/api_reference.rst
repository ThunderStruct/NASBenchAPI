API Reference
=============

This document provides comprehensive reference for NASBench-101/201/301 APIs, including architecture representations, method signatures, return types, and benchmark-specific details.

Benchmarks Overview
-------------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 30 15 15

   * - Benchmark
     - Datasets
     - Available Splits
     - Primary Metrics
     - Training Epochs
   * - NASBench-101
     - CIFAR-10
     - train, val, test
     - train/val/test accuracy
     - 4, 12, 36, 108
   * - NASBench-201
     - CIFAR-10, CIFAR-100, ImageNet16-120
     - train, val, test
     - train/val/test accuracy, losses
     - 0-199 (200 epochs total)
   * - NASBench-301
     - CIFAR-10, CIFAR-100
     - val, test
     - surrogate val/test accuracy
     - N/A (surrogate-based)

Architecture Representations
----------------------------

Each benchmark uses a different architecture representation:

**NASBench-101 (Arch101):**

- Dataclass with two fields:

  - ``adjacency``: list[list[int]] — 7×7 adjacency matrix
  - ``operations``: list[str] — 7 operations from ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'output']

- Example:

.. code-block:: python

   Arch101(
       adjacency=[[0, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0],
                  ...],
       operations=['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', ..., 'output']
   )

**NASBench-201 (String):**

- Architecture string format: ``|op~0|+|op~0|op~1|+|op~0|op~1|op~2|``
- 6 edges connecting 4 nodes in a cell
- 5 operations per edge: ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
- Total search space: 5^6 = 15,625 unique architectures
- Each architecture maps to a canonical index (0-15624)

- Example:

.. code-block:: python

   '|none~0|+|skip_connect~0|nor_conv_1x1~1|+|nor_conv_3x3~0|avg_pool_3x3~1|skip_connect~2|'

**NASBench-301 (Dict):**

- DARTS-style architecture with normal and reduction cells
- Dictionary with 'normal' and 'reduce' keys
- Each cell: list of (operation, predecessor_node) tuples
- 8 operations: ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'none']
- 4 intermediate nodes per cell, each with 2 input edges

- Example:

.. code-block:: python

   {
       'normal': [('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ...],
       'reduce': [('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ...]
   }

Common API Surface
------------------

All benchmarks expose the following core methods.

Initialization
~~~~~~~~~~~~~~

.. code-block:: python

   from nasbenchapi import NASBench101, NASBench201, NASBench301

   # Using explicit path
   api = NASBench201('/path/to/nb201.pkl', verbose=True)

   # Using environment variable
   api = NASBench201(verbose=True)  # Reads from NASBENCH201_PATH

**Constructor Args:**

- ``pickle_path``: Optional[str] — path to pickled benchmark data; if None, reads from environment variable
- ``verbose``: bool — enable/disable all logging output (default: True)

**Environment Variables:**

- ``NASBENCH101_PATH`` — path to NB101 pickle file
- ``NASBENCH201_PATH`` — path to NB201 pickle file
- ``NASBENCH301_PATH`` — path to NB301 pickle file

get_statistics
~~~~~~~~~~~~~~

Get statistics about the loaded benchmark data.

.. code-block:: python

   stats = api.get_statistics()

**Returns:** dict — benchmark statistics

**Return Format by Benchmark:**

- NB101: ``{'benchmark': 'nasbench101', 'architectures': int, 'records': int}``
- NB201: ``{'benchmark': 'nasbench201', 'entries': int}``
- NB301: ``{'benchmark': 'nasbench301', 'files': int}``

random_sample
~~~~~~~~~~~~~

Sample random architectures from the benchmark search space.

.. code-block:: python

   samples = api.random_sample(n=5, seed=123)

**Args:**

- ``n``: int — number of samples (default: 1)
- ``seed``: Optional[int] — RNG seed for reproducibility

**Returns:**

- **NB101**: list[Arch101] — list of Arch101 dataclass objects
- **NB201**: list[str] — list of architecture strings
- **NB301**: list[int] — indices for entries in the loaded dataset (falls back to synthetic architecture dicts if raw entries are unavailable)

iter_all
~~~~~~~~

Iterate over all available architectures in the loaded data.

.. code-block:: python

   for arch in api.iter_all():
       result = api.query(arch, dataset='cifar10', split='val')

**Returns:**

- **NB101**: Iterator[Arch101]
- **NB201**: Iterator[str] — architecture strings
- **NB301**: Iterator[int] — indices in loaded data

get_index
~~~~~~~~~

Get an identifier or index for an architecture.

.. code-block:: python

   # NB201: Convert arch string to numeric index
   idx = api.get_index('|none~0|+|skip_connect~0|nor_conv_1x1~1|+|...')
   # Returns: 12345 (int in range 0-15624)

   # NB101: Get hash identifier
   hash_id = api.get_index(arch_obj)
   # Returns: 'a3f5b2...' (SHA256 hash string)

   # NB301: Find index in loaded data
   idx = api.get_index(arch_dict)
   # Returns: 42 or None

**Args:**

- ``arch``: Architecture representation (type depends on benchmark)

  - NB101: Arch101 object
  - NB201: str (architecture string)
  - NB301: dict (architecture dict)

**Returns:**

- **NB101**: str — stable SHA256 hash identifier
- **NB201**: int — canonical index (0-15624)
- **NB301**: Optional[int] — index in loaded data, or None if not found

available_budgets
~~~~~~~~~~~~~~~~~

List available training budgets (epochs) for a dataset/split combination.

.. code-block:: python

   budgets = api.available_budgets(dataset='cifar10', split='val')
   # Returns e.g. [199, 200] for NB201 validation

**Args:**

- ``dataset``: Optional[str] — target dataset (defaults to all datasets)
- ``split``: Optional[str] — target split (defaults to all splits)

**Returns:** Optional[list] — sorted list of budgets if tracked; None when budgets are not defined for the benchmark.

- **NB101**: returns ``None`` (budgets not tracked)
- **NB201**: list of available epochs per dataset/split based on original training logs
- **NB301**: epochs derived from per-entry learning curves (validation) or final declared budget (test)

exists
~~~~~~

Validate whether a combination of dataset, split, budget, and architecture is supported without issuing a full ``query``.

.. code-block:: python

   api.exists(dataset='cifar10', split='val', budget=199)  # -> True

**Args:**

- ``dataset``: Optional[str]
- ``split``: Optional[str]
- ``budget``: Optional[Any]
- ``arch``: Optional[Any] — architecture representation

**Returns:** bool — True if every provided component is supported, False otherwise.

query
~~~~~

Query performance metrics for an architecture from loaded data.

.. code-block:: python

   # NB201 example
   result = api.query(
       arch='|none~0|+|skip_connect~0|nor_conv_1x1~1|+|...',
       dataset='cifar10',
       split='val',
       seed=777,
       budget=199
   )
   print(f"Validation accuracy: {result['metric']:.2f}%")
   print(f"Training time: {result['cost']:.2f}s")

**Args:**

- ``arch``: Architecture representation (depends on benchmark)

  - **NB101**: Arch101 object
  - **NB201**: str (architecture string)
  - **NB301**: Any (dict or index)

- ``dataset``: str — dataset name

  - **NB101**: 'cifar10'
  - **NB201**: 'cifar10', 'cifar100', 'ImageNet16-120'
  - **NB301**: 'cifar10', 'cifar100'

- ``split``: str — data split

  - **NB101**: 'train', 'val', 'test'
  - **NB201**: 'train', 'val', 'test'
  - **NB301**: 'val', 'test'

- ``seed``: Optional[int] — random seed (default varies by benchmark)

  - **NB201**: default 777 (official NB201 seed)
  - **NB101/NB301**: unused

- ``budget``: Optional[Any] — training budget

  - **NB101**: unused (returns final recorded metrics)
  - **NB201**: epoch number 0-199 (default: 199 for final epoch)
  - **NB301**: epoch index for validation curves (defaults to final epoch); test split always reports the declared final budget

**Returns:** dict with the following keys:

.. code-block:: python

   {
       'metric': Optional[float],      # Primary metric (e.g., accuracy %)
       'metric_name': str,              # Name of metric (e.g., 'val_acc')
       'cost': Optional[float],         # Training time in seconds
       'std': Optional[float],          # Standard deviation (if available)
       'info': dict                     # Additional metadata and raw data
   }

**Return Value Details:**

- ``metric``: Accuracy percentage (e.g., 94.5) or None if not available
- ``metric_name``: Describes the metric, typically '{split}_acc'
- ``cost``: Training/evaluation time in seconds, or None
- ``std``: Standard deviation of the metric across multiple runs (rarely used)
- ``info``: Dictionary containing additional information:

  - **NB201**: arch_index, dataset, split, seed, epoch, arch_str, params, flop
  - **NB101**: Full raw record from the benchmark
  - **NB301**: Entry metadata (index, dataset, epochs available/used, declared budget, optimizer tag, JSON path)

NASBench-101 Specifics
------------------------

Import and Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nasbenchapi import NASBench101

   api = NASBench101('/path/to/nasbench_only108.pkl', verbose=True)
   # Or use environment variable
   api = NASBench101(verbose=True)

Dataset and Splits
~~~~~~~~~~~~~~~~~~

- **Single dataset**: CIFAR-10 only
- **Splits**: train, val, test
- **Training epochs**: 4, 12, 36, 108 (typically query final epoch 108)

Architecture Type (Arch101)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nasbenchapi import Arch101

   arch = Arch101(
       adjacency=[[0, 1, 1, 0, 0, 0, 0], ...],  # 7×7 matrix
       operations=['input', 'conv3x3-bn-relu', ..., 'output']  # 7 ops
   )

Operations
~~~~~~~~~~

Available operations (from ``op_set()``):

- 'input' (fixed at node 0)
- 'conv3x3-bn-relu'
- 'conv1x1-bn-relu'
- 'maxpool3x3'
- 'output' (fixed at node 6)

encode / decode / id
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Encode Arch101 to native strings
   encoding = api.encode(arch)
   # Returns: {'adjacency_str': '0110000...', 'operations_str': 'input,conv3x3-bn-relu,...'}

   # Decode encoding to Arch101
   arch = api.decode(encoding)

   # Get stable hash ID
   arch_id = api.id(arch)
   # Returns: 'a3f5b2c8...' (SHA256 hash)

get_index
~~~~~~~~~

.. code-block:: python

   # Returns the same as id() for consistency
   hash_id = api.get_index(arch)
   # Returns: 'a3f5b2c8...'

random_sample
~~~~~~~~~~~~~

.. code-block:: python

   archs = api.random_sample(n=10, seed=42)
   # Returns: list of 10 Arch101 objects sampled from loaded data

iter_all
~~~~~~~~

.. code-block:: python

   for arch in api.iter_all():
       result = api.query(arch, dataset='cifar10', split='test')
       print(f"Test acc: {result['metric']:.2f}%")

query
~~~~~

.. code-block:: python

   result = api.query(arch, dataset='cifar10', split='val')

**Args:**

- ``arch``: Arch101 — architecture object
- ``dataset``: str — 'cifar10' (only dataset available)
- ``split``: str — 'train', 'val', or 'test'
- ``seed``: Optional[int] — unused
- ``budget``: Optional[Any] — unused (always uses final epoch data)

**Returns:** dict with keys: metric, metric_name, cost, std, info

train_time
~~~~~~~~~~

Get training time for an architecture.

.. code-block:: python

   time_sec = api.train_time(arch, dataset='cifar10')
   # Returns: float (seconds) or None

mutate
~~~~~~

Apply a mutation to an architecture.

.. code-block:: python

   import random
   rng = random.Random(42)
   mutated = api.mutate(arch, rng=rng, kind='edge_toggle')

**Mutation kinds:**

- 'edge_toggle' — flip an edge in the adjacency matrix
- 'op_swap' — swap two operations

NASBench-201 Specifics
------------------------

Import and Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nasbenchapi import NASBench201

   api = NASBench201('/path/to/NASBench-201-v1_1-096897.pth', verbose=True)
   # Or use environment variable
   api = NASBench201(verbose=True)

Dataset and Splits
~~~~~~~~~~~~~~~~~~

- **Datasets**: CIFAR-10, CIFAR-100, ImageNet16-120
- **Splits**: train, val, test
- **Training epochs**: 0-199 (200 epochs total)
- **Common budget values**: 12 (early), 199 (final epoch)
- **Default seed**: 777 (official NB201 seed)

Architecture Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NB201 uses **architecture strings** as the primary representation:

.. code-block:: python

   arch_str = '|none~0|+|skip_connect~0|nor_conv_1x1~1|+|nor_conv_3x3~0|avg_pool_3x3~1|skip_connect~2|'

**Format details:**

- Cell with 4 nodes (node 0 is input, nodes 1-3 are intermediate, node 4 is output)
- 6 edges: (1←0), (2←0), (2←1), (3←0), (3←1), (3←2)
- Each edge has one operation from: ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
- String format: ``|op~src|+|op~src|op~src|+|op~src|op~src|op~src|``

**Index mapping:**

- Each architecture has a canonical integer index: 0 to 15,624
- Use ``get_index(arch_str)`` to convert string → index

random_sample
~~~~~~~~~~~~~

.. code-block:: python

   arch_strs = api.random_sample(n=5, seed=42)
   # Returns: ['|none~0|+|...', '|skip_connect~0|+|...', ...]

**Returns**: list[str] — architecture strings

iter_all
~~~~~~~~

.. code-block:: python

   for arch_str in api.iter_all():
       idx = api.get_index(arch_str)
       print(f"Architecture {idx}: {arch_str}")

**Returns**: Iterator[str] — yields architecture strings

get_index
~~~~~~~~~

Convert an architecture string to its canonical integer index.

.. code-block:: python

   idx = api.get_index('|none~0|+|skip_connect~0|nor_conv_1x1~1|+|...')
   # Returns: 12345 (int in range 0-15624)

**Args:**

- ``arch``: str — NB201 architecture string

**Returns:** int — index (0-15624)

**Raises:** ValueError if architecture string is invalid

query
~~~~~

.. code-block:: python

   result = api.query(
       arch='|none~0|+|skip_connect~0|nor_conv_1x1~1|+|...',
       dataset='cifar10',
       split='val',
       seed=777,      # Default seed
       budget=199     # Final epoch
   )

**Args:**

- ``arch``: str — NB201 architecture string
- ``dataset``: str — 'cifar10', 'cifar100', or 'ImageNet16-120'
- ``split``: str — 'train', 'val', or 'test'
- ``seed``: Optional[int] — data seed (default: 777)
- ``budget``: Optional[int] — epoch number 0-199 (default: 199)

**Returns:** dict with keys:

- ``metric``: accuracy percentage (e.g., 91.23)
- ``metric_name``: '{split}_acc'
- ``cost``: training/eval time in seconds
- ``std``: None (not used)
- ``info``: dict with arch_index, dataset, split, seed, epoch, arch_str, params, flop

**Split-specific behavior:**

- 'train': Returns training accuracy at specified epoch
- 'val': Returns validation accuracy (uses 'x-valid@epoch' keys in data)
- 'test': Returns test accuracy (uses 'ori-test@epoch' keys, falls back to validation)

NASBench-301 Specifics
------------------------

Import and Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nasbenchapi import NASBench301

   api = NASBench301('/path/to/nb301_data.pkl', verbose=True)
   # Or use environment variable
   api = NASBench301(verbose=True)

Dataset and Splits
~~~~~~~~~~~~~~~~~~

- **Datasets**: CIFAR-10, CIFAR-100
- **Splits**: val, test (no train split for surrogates)
- **Training epochs**: Validation learning curves provide per-epoch accuracies; the test split reports metrics at the declared final budget for each entry.

Architecture Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NB301 uses **DARTS-style architecture dictionaries**:

.. code-block:: python

   arch = {
       'normal': [
           ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),  # Node 1 inputs
           ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),  # Node 2 inputs
           ('sep_conv_3x3', 1), ('skip_connect', 0),  # Node 3 inputs
           ('skip_connect', 0), ('dil_conv_3x3', 2)   # Node 4 inputs
       ],
       'reduce': [
           ('max_pool_3x3', 0), ('max_pool_3x3', 1),
           ('skip_connect', 2), ('max_pool_3x3', 0),
           ('max_pool_3x3', 0), ('skip_connect', 2),
           ('skip_connect', 2), ('max_pool_3x3', 1)
       ]
   }

**Format details:**

- Two cells: 'normal' and 'reduce' (reduction cell)
- Each cell has 4 intermediate nodes
- Each node selects 2 operations from previous nodes (including input nodes 0 and 1)
- 8 operations: ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'none']
- Each entry is a tuple: (operation_name, predecessor_node_index)

random_sample
~~~~~~~~~~~~~

.. code-block:: python

   indices = api.random_sample(n=3, seed=42)
   # Returns: [102, 4096, 7123]

**Returns**: list[int] — dataset entry indices (falls back to architecture dict samples if raw entries are unavailable)

iter_all
~~~~~~~~

.. code-block:: python

   for idx in api.iter_all():
       print(f"Architecture index: {idx}")

**Returns**: Iterator[int] — yields indices in loaded data

get_index
~~~~~~~~~

Find the index of an architecture in loaded data.

.. code-block:: python

   idx = api.get_index(arch_dict)
   # Returns: 42 (int) or None if not found

**Args:**

- ``arch``: Any — architecture dict, dataset index, or entry path string

**Returns:** Optional[int] — index in loaded data, or None if not found

query
~~~~~

.. code-block:: python

   result = api.query(
       arch=0,           # dataset index
       dataset='cifar10',
       split='val',
       budget=50,        # epoch index
   )

**Args:**

- ``arch``: Any — dataset index (int), entry path (str), or architecture dict with 'normal'/'reduce' keys
- ``dataset``: str — 'cifar10' or 'cifar100'
- ``split``: str — 'val' or 'test'
- ``seed``: Optional[int] — unused
- ``budget``: Optional[int] — epoch index for validation curves (defaults to final epoch); ignored for test split

**Returns:** dict with keys: metric, metric_name, cost, std, info (runtime in seconds, dataset metadata, epochs available/used, declared budget, optimizer tag, and JSON path)

**Split behavior:**

- ``val``: accuracy from the per-entry validation learning curve; budgets beyond the recorded length fall back to the final epoch.
- ``test``: reported test accuracy at the declared final budget (the ``budget`` argument is ignored).

Complete Usage Examples
-----------------------

NASBench-101 Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nasbenchapi import NASBench101

   # Initialize
   api = NASBench101(verbose=True)
   stats = api.get_statistics()
   print(f"Loaded {stats['architectures']} architectures")

   # Sample architectures
   archs = api.random_sample(n=5, seed=42)

   # Query performance
   for arch in archs:
       result = api.query(arch, dataset='cifar10', split='test')
       print(f"Test accuracy: {result['metric']:.2f}%")
       print(f"Training time: {result['cost']:.2f}s")

NASBench-201 Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nasbenchapi import NASBench201

   # Initialize
   api = NASBench201(verbose=True)

   # Sample architecture strings
   arch_strs = api.random_sample(n=3, seed=777)

   # Query on multiple datasets
   for arch_str in arch_strs:
       idx = api.get_index(arch_str)
       print(f"\nArchitecture {idx}:")

       for dataset in ['cifar10', 'cifar100', 'ImageNet16-120']:
           result = api.query(
               arch=arch_str,
               dataset=dataset,
               split='test',
               seed=777,
               budget=199
           )
           print(f"  {dataset} test acc: {result['metric']:.2f}%")

   # Iterate all architectures
   count = 0
   for arch_str in api.iter_all():
       count += 1
       if count > 5:
           break
       result = api.query(arch_str, dataset='cifar10', split='val')
       print(f"Arch {count}: val_acc = {result['metric']:.2f}%")

NASBench-301 Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nasbenchapi import NASBench301

   # Initialize
   api = NASBench301(verbose=True)

   # Sample dataset indices
   arch_indices = api.random_sample(n=2, seed=42)

   # Query performance at multiple epochs
   for idx in arch_indices:
       final_val = api.query(idx, dataset='cifar10', split='val')
       mid_val = api.query(idx, dataset='cifar10', split='val', budget=50)
       print(f"Index {idx}: final={final_val['metric']:.2f}% | mid@50={mid_val['metric']:.2f}%")

Error Handling
--------------

Common Exceptions
~~~~~~~~~~~~~~~~~

**ValueError**:

- Invalid architecture string format (NB201)
- Architecture index out of range
- Invalid dataset or split name

**FileNotFoundError**:

- Pickle file not found at specified path
- Environment variable not set

**KeyError**:

- Data format mismatch (e.g., missing expected keys in pickle)

Example Error Handling
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nasbenchapi import NASBench201

   try:
       api = NASBench201('/path/to/data.pkl', verbose=True)
   except FileNotFoundError:
       print("Data file not found. Please set NASBENCH201_PATH or provide valid path.")
       exit(1)

   try:
       result = api.query(
           arch='|invalid~format|',
           dataset='cifar10',
           split='val'
       )
   except ValueError as e:
       print(f"Invalid architecture: {e}")

Verbose Logging Control
-----------------------

All benchmarks support a ``verbose`` parameter to control logging output:

.. code-block:: python

   # Enable all logging (default)
   api = NASBench201(verbose=True)
   # Outputs:
   # Loading NB201 from /path/to/file.pkl (2.1 GB)
   # Reading: 100%|██████████| 2.1G/2.1G [00:15<00:00]
   # Unpickling data...
   # Unpickling complete.
   # [NB201] Loaded 15625 architectures (source=arch2infos)

   # Disable all logging (silent mode)
   api = NASBench201(verbose=False)
   # No output

Logging includes:

- File loading progress bars (via tqdm)
- Unpickling status messages
- Data summary and statistics
- Warning messages (e.g., mapping failures)
