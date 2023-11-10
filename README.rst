===========
pad-channel
===========

Encoding padding statuses as an input channel for enhanced performance in
convolutional neural networks

All the commands below are run in terminal at the project root directory.

Setup
-----

.. code-block:: sh

   git submodule update --init --recursive
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

Train with and without PadChannel
---------------------------------

Train ``resnet50`` with and without PadChannel on the ``ImageNet-1K`` dataset.

.. code-block:: sh

   python submodules/torchvision/references/classification/train.py \
       --model resnet50 \
       --lr 0.0125 \
       --epochs 100 \
       --data-path /path/to/dataset \
       --output-dir /path/to/output \
       --pad-channel

   python submodules/torchvision/references/classification/train.py \
       --model resnet50 \
       --lr 0.0125 \
       --epochs 100 \
       --data-path /path/to/dataset \
       --output-dir /path/to/output

Train ``vgg16_bn`` with and without PadChannel on the ``ImageNet-1K`` dataset.

.. code-block:: sh

   python submodules/torchvision/references/classification/train.py \
       --model vgg16_bn \
       --lr 0.0125 \
       --epochs 100 \
       --data-path /path/to/dataset \
       --output-dir /path/to/output \
       --pad-channel

   python submodules/torchvision/references/classification/train.py \
       --model vgg16_bn \
       --lr 0.0125 \
       --epochs 100 \
       --data-path /path/to/dataset \
       --output-dir /path/to/output

Evaluate Models without PadChannel with Pre-trained Weights
-----------------------------------------------------------

Evaluate ``resnet50`` with and without PadChannel on the ``ImageNet-1K``
dataset.

.. code-block:: sh

   python submodules/torchvision/references/classification/train.py \
       --model resnet50 \
       --resume /path/to/checkpoint \
       --data-path /path/to/dataset \
       --test-only \
       --pad-channel

   python submodules/torchvision/references/classification/train.py \
       --model resnet50 \
       --resume /path/to/checkpoint \
       --data-path /path/to/dataset \
       --test-only

Evaluate ``vgg16_bn`` with and without PadChannel on the ``ImageNet-1K``
dataset.

.. code-block:: sh

   python submodules/torchvision/references/classification/train.py \
       --model vgg16_bn \
       --resume /path/to/checkpoint \
       --data-path /path/to/dataset \
       --test-only \
       --pad-channel

   python submodules/torchvision/references/classification/train.py \
       --model vgg16_bn \
       --resume /path/to/checkpoint \
       --data-path /path/to/dataset \
       --test-only
