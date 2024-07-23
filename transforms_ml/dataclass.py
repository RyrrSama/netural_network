# Dependency import
import torch


class TextData:
    def __init__(self, train_data, test_data, block_size, batch_size):
        self.train_data = train_data
        self.test_data = test_data
        self.block_size = block_size
        self.batch_size = batch_size

    def get_context_target(self):
        x_train = self.train_data[: self.block_size]
        y_train = self.train_data[1 : self.block_size + 1]

        for char_index in range(self.block_size):
            context = x_train[: char_index + 1]
            target = y_train[char_index]
            print(f"The context is {context} and the target : {target}")

    def get_batch(self, split):
        # generate a small batch of data of input x and target y
        data = self.train_data if split == "train" else self.test_data
        context_index_list = torch.randint(
            len(data) - self.block_size, (self.batch_size,)
        )
        x_train_batch_data = torch.stack(
            [
                data[context_index : context_index + self.block_size]
                for context_index in context_index_list
            ]
        )
        y_train_batch_data = torch.stack(
            [
                data[context_index + 1 : context_index + self.block_size + 1]
                for context_index in context_index_list
            ]
        )
        return x_train_batch_data, y_train_batch_data
