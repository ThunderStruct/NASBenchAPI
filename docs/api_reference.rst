API Reference
=============

This document summarizes the unified API surface for NAS-Bench-101/201/301, including method usage, arguments, return values, and benchmark metadata (datasets, epochs, metrics).

Benchmarks Overview
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 30 20

   * - Benchmark
     - Datasets
     - Primary Metrics
     - Training Epochs
   * - NAS-Bench-101
     - CIFAR-10
     - train/val/test accuracy
     - 4, 12, 36, 108
   * - NAS-Bench-201
     - CIFAR-10, CIFAR-100, ImageNet16-120
     - train/val/test accuracy, losses
     - 12, 200
   * - NAS-Bench-301
     - CIFAR-10, CIFAR-100
     - surrogate val/test accuracy
     - N/A (surrogate)

.. note::
   NAS-Bench-301 relies on surrogate models; epochs are not applicable.

Common API Surface
------------------

All benchmarks expose the following methods via ``nasbenchapi.nb_api`` classes: ``NASBench101``, ``NASBench201``, and ``NASBench301``.

load
~~~~

Load the benchmark dataset.

.. code-block:: python

   api = NASBench101().load('/path/to/nb101.pkl')

**Args:**

- ``data_path``: Optional[str] — explicit dataset path; if omitted, env var is used.

**Returns:** self instance

.. note::
   Environment variables are set as follows:

   - ``NASBENCH101_PATH``
   - ``NASBENCH201_PATH``
   - ``NASBENCH301_PATH``

close
~~~~~

Close and cleanup resources.

.. code-block:: python

   api.close()

**Args:** none

**Returns:** None

set_seed
~~~~~~~~

Set the random seed for reproducibility.

.. code-block:: python

   api.set_seed(42)

**Args:**

- ``seed``: int — RNG seed

**Returns:** None

bench_name
~~~~~~~~~~

Get the benchmark identifier.

.. code-block:: python

   name = api.bench_name()

**Args:** none

**Returns:** str — short name (e.g., 'nb101', 'nb201', 'nb301')

datasets
~~~~~~~~

List available datasets.

.. code-block:: python

   ds = api.datasets()

**Args:** none

**Returns:** list[str] — dataset names

splits
~~~~~~

Get data splits for a dataset.

.. code-block:: python

   spl = api.splits('cifar10')

**Args:**

- ``dataset``: str — dataset name

**Returns:** list[str] — supported splits (['train', 'val', 'test'])

decode
~~~~~~

Decode a benchmark-native encoding to an architecture object.

.. code-block:: python

   arch = api.decode(encoding)

**Args:**

- ``encoding``: Any — benchmark-native representation

**Returns:** Any — decoded architecture object

encode
~~~~~~

Encode an architecture object to benchmark-native representation.

.. code-block:: python

   encoding = api.encode(arch)

**Args:**

- ``arch``: Any — architecture object

**Returns:** Any — benchmark-native encoding

id
~~

Get a stable identifier for an architecture.

.. code-block:: python

   arch_id = api.id(arch)

**Args:**

- ``arch``: Any — architecture object

**Returns:** str — stable identifier

random_sample
~~~~~~~~~~~~~

Sample random architectures from the benchmark.

.. code-block:: python

   samples = api.random_sample(n=5, seed=123)

**Args:**

- ``n``: int — number of samples (default 1)
- ``seed``: Optional[int] — RNG seed

**Returns:** list[Any] — sampled architectures

iter_all
~~~~~~~~

Iterate over all available architectures (if supported).

.. code-block:: python

   for arch in api.iter_all():
       ...

**Args:** none

**Returns:** Iterator[Any] — iterate all available architectures

mutate
~~~~~~

Mutate an architecture.

.. code-block:: python

   mut = api.mutate(arch, rng=random.Random(0), kind='edge_toggle')

**Args:**

- ``arch``: Any — architecture to mutate
- ``rng``: random.Random — RNG instance
- ``kind``: Optional[str] — mutation kind (benchmark-defined)

**Returns:** Any — mutated architecture

NAS-Bench-101 Specifics
------------------------

Import
~~~~~~

.. code-block:: python

   from nasbenchapi import NASBench101
   api = NASBench101('/path/to/nb101.pkl')

Architecture Type (Arch101)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``adjacency``: list[list[int]] — 7x7 adjacency matrix
- ``operations``: list[str] — 7 operations

op_set
~~~~~~

Get available operations.

.. code-block:: python

   ops = api.op_set()

**Args:** none

**Returns:** list[str] — available operations (e.g., input, conv3x3-bn-relu, ...)

query
~~~~~

Query performance metrics for an architecture from loaded data.

.. code-block:: python

   res = api.query(arch, dataset='cifar10', split='val')

**Args:**

- ``arch``: Arch101 — architecture to query
- ``dataset``: str — dataset name ('cifar10')
- ``split``: str — 'train', 'val', or 'test'
- ``seed``: Optional[int] — reproducibility seed (unused)
- ``budget``: Optional[Any] — optional budget (unused)

**Returns:** dict — keys:

- ``metric``: float | None — accuracy for the specified split
- ``metric_name``: str — e.g., 'val_acc', 'test_acc', 'train_acc'
- ``cost``: float | None — training time in seconds
- ``std``: float | None — standard deviation (not used in NB101)
- ``info``: dict — raw record data

train_time
~~~~~~~~~~

Get training time for an architecture.

.. code-block:: python

   t = api.train_time(arch, dataset='cifar10')

**Args:**

- ``arch``: Arch101 — architecture
- ``dataset``: str — dataset name

**Returns:** float | None — training time if available

Encoding/Decoding Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Encode
   enc = api.encode(arch)
   # Decode
   arch2 = api.decode(enc)
   # Stable ID
   hid = api.id(arch)

NAS-Bench-201 Specifics
------------------------

Import
~~~~~~

.. code-block:: python

   from nasbenchapi import NASBench201
   api = NASBench201('/path/to/nb201.pkl')

**Datasets:** ['cifar10', 'cifar100', 'ImageNet16-120']

Architecture Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NB201 uses a cell-based search space with:

- 6 edges connecting 4 nodes
- 5 possible operations per edge: ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
- Total search space: 5^6 = 15,625 architectures
- **Architectures identified by integer indices (0-15624)**
  with a canonical mapping to/from NB201 arch strings

Official NB201 format stores architecture strings in metadata: ``|op~0|+|op~0|op~1|+|op~0|op~1|op~2|``

random_sample
~~~~~~~~~~~~~

Sample random architectures from the search space or loaded data.

.. code-block:: python

   samples = api.random_sample(n=5, seed=42)
   # Returns list of integer indices: [0, 453, 8921, ...]

- If data is loaded: samples from available architecture indices (0-15624)
- If no data: samples uniformly from the full index range

random_sample_str
~~~~~~~~~~~~~~~~~

Sample random architectures as NB201 arch strings.

.. code-block:: python

   samples = api.random_sample_str(n=3, seed=7)
   # Returns list of strings: ['|op~0|+|op~0|op~1|+|op~0|op~1|op~2|', ...]

- Uses loaded mappings when available; otherwise derives from index

iter_all
~~~~~~~~

Iterate over all architectures in loaded data.

.. code-block:: python

   for arch_idx in api.iter_all():
       result = api.query(arch_idx, dataset='cifar10', split='val')
       print(f"Arch {arch_idx}: {result['metric']}")

query
~~~~~

Query performance metrics for an architecture.

.. code-block:: python

   # Accepts either index or arch string
   result = api.query(0, dataset='cifar10', split='val', budget=199)
   result2 = api.query('|nor_conv_3x3~0|+|skip_connect~0|nor_conv_1x1~1|+|avg_pool_3x3~0|none~1|skip_connect~2|',
                       dataset='cifar10', split='val', budget=199)
   print(f"Validation accuracy: {result['metric']:.2f}%")

**Args:**

- ``arch``: int | str — architecture index (0-15624) or NB201 arch string
- ``dataset``: str — 'cifar10', 'cifar100', or 'ImageNet16-120'
- ``split``: str — 'train', 'val', or 'test'
- ``seed``: Optional[int] — data seed (default: 777)
- ``budget``: Optional[int] — epoch number 0-199 (default: 199 for final epoch)

**Returns:** dict with 'metric', 'metric_name', 'cost', 'std', 'info'

Conversions
~~~~~~~~~~~

Helper methods to convert between indices and arch strings:

.. code-block:: python

   s = api.index_to_arch_str(123)
   i = api.arch_str_to_index(s)

Current Behavior
~~~~~~~~~~~~~~~~

- ``decode/encode/id``: pass-through placeholders until canonical representation is defined
- ``random_sample/iter_all``: implemented; samples indices by default
- ``random_sample_str``: implemented; samples arch strings
- ``query``: accepts index or arch string and returns real metrics from loaded data
- ``mutate``: no-op

NAS-Bench-301 Specifics
------------------------

Import
~~~~~~

.. code-block:: python

   from nasbenchapi import NASBench301
   api = NASBench301('/path/to/nb301.pkl')

**Datasets:** ['cifar10', 'cifar100']

Architecture Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NB301 uses DARTS-style search space with:

- 8 operations: ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'none']
- 4 intermediate nodes per cell
- Each node selects 2 operations from previous nodes
- Two cells: normal and reduction

Architecture representation: dict with 'normal' and 'reduce' keys, each containing a list of (operation, predecessor) tuples.

random_sample
~~~~~~~~~~~~~

Sample random architectures from the search space or loaded data.

.. code-block:: python

   samples = api.random_sample(n=3, seed=42)
   # Returns list of dicts: {'normal': [...], 'reduce': [...]}

- If data is loaded: samples from available architectures
- If no data: generates random DARTS-style architectures from search space

iter_all
~~~~~~~~

Iterate over all architectures in loaded data.

.. code-block:: python

   for arch in api.iter_all():
       print(arch)

query
~~~~~

Query performance metrics for an architecture.

.. code-block:: python

   result = api.query(arch_dict, dataset='cifar10', split='val')
   # Note: Returns placeholder without surrogate models

**Args:**

- ``arch``: Any — architecture representation (dict or index)
- ``dataset``: str — 'cifar10' or 'cifar100'
- ``split``: str — 'val' or 'test'
- ``seed``: Optional[int] — reproducibility seed (unused)
- ``budget``: Optional[Any] — budget specification (unused)

**Returns:** dict with 'metric', 'metric_name', 'cost', 'std', 'info'

**Note:** NB301 requires surrogate models for predictions. Currently returns placeholder values.

Current Behavior
~~~~~~~~~~~~~~~~

- ``decode/encode/id``: pass-through placeholders
- ``random_sample/iter_all``: **fully implemented** - samples from data or generates from search space
- ``query``: **placeholder** - requires surrogate model integration for real predictions
- ``mutate``: no-op
