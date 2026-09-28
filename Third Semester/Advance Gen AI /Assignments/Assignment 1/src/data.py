import os
import numpy as np
import torch


def load_token_array(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    file_size = os.path.getsize(path)

    if file_size % 2 != 0:
        raise ValueError("file size must be even")

    return np.memmap(
        path,
        mode="r",
        dtype=np.dtype("<u2"),
    )
    
def get_batch(

    dataset,

    batch_size,

    sequence_length,

    device,

    generator,

):

    if batch_size <= 0:

        raise ValueError("batch_size must be positive")

    if sequence_length <= 0:

        raise ValueError("sequence_length must be positive")

    if len(dataset) < sequence_length + 1:

        raise ValueError(

            "dataset does not contain enough tokens"

        )

    max_start = len(dataset) - sequence_length

    starts = torch.randint(

        0,

        max_start,

        (batch_size,),

        generator=generator,

    )

    inputs = []

    targets = []

    for start in starts.tolist():

        window = dataset[

            start:start + sequence_length + 1

        ]

        inputs.append(window[:-1])

        targets.append(window[1:])

    inputs = torch.tensor(

        np.stack(inputs),

        dtype=torch.long,

        device=device,

    )

    targets = torch.tensor(

        np.stack(targets),

        dtype=torch.long,

        device=device,

    )

    return inputs, targets