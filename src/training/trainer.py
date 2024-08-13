import torch
import torch.nn as nn
import torch.optim as optim
import time


def decode(x, tokenizer):
    x = torch.softmax(x, -1)
    x = torch.argmax(x, dim=-1)
    x = x.tolist()
    print("argmax x:", x)
    return tokenizer.decode(x)


def train_model(
    encoder_decoder_transformer,
    train_dataloader,
    val_dataloader,
    tgt_tokenizer,
    device,
    state_dict_filename,
):
    torch.manual_seed(25)

    epochs = 1000
    print(encoder_decoder_transformer.parameters())
    opt = optim.AdamW(
        encoder_decoder_transformer.parameters(), lr=0.0001, weight_decay=0.0001
    )
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=2)
    training_step = 0
    validation_step = 0
    best_val_loss = 100
    num_fails = 0

    # Large models need this to actually train
    for p in encoder_decoder_transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for i in range(epochs):
        epoch_time_start = time.time()
        dataloader_iter = iter(train_dataloader)
        train_losses = []
        val_losses = []
        for src_batch, tgt_batch in dataloader_iter:
            src_tokens = torch.IntTensor([sequence.ids for sequence in src_batch]).to(
                device
            )
            encoder_input = src_tokens
            train_tgt_tokens = torch.IntTensor(
                [sequence.ids for sequence in tgt_batch]
            ).to(device)
            decoder_input = train_tgt_tokens[:, :-1]
            decoder_desired_output_train = train_tgt_tokens[:, 1:]
            src_masks = torch.IntTensor(
                [sequence.attention_mask for sequence in src_batch]
            ).to(device)
            tgt_masks = torch.IntTensor(
                [sequence.attention_mask for sequence in tgt_batch]
            )[:, :-1].to(device)
            train_output = encoder_decoder_transformer(
                encoder_input, decoder_input, src_masks, tgt_masks
            )
            output_transpose = train_output.transpose(
                -1, -2
            )  # output needs to be N, C, other dimension for torch cross entropy
            loss = loss_function(output_transpose, decoder_desired_output_train.long())
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
            if training_step % 20 == 0:
                print("Completed training step", training_step)
            training_step += 1

        for src_batch, tgt_batch in val_dataloader:
            src_tokens = torch.IntTensor([sequence.ids for sequence in src_batch]).to(
                device
            )
            encoder_input = src_tokens
            val_tgt_tokens = torch.IntTensor(
                [sequence.ids for sequence in tgt_batch]
            ).to(device)
            decoder_input = val_tgt_tokens[:, :-1]
            decoder_desired_output_val = val_tgt_tokens[:, 1:]
            src_masks = torch.IntTensor(
                [sequence.attention_mask for sequence in src_batch]
            ).to(device)
            tgt_masks = torch.IntTensor(
                [sequence.attention_mask for sequence in tgt_batch]
            )[:, :-1].to(device)
            val_output = encoder_decoder_transformer(
                encoder_input, decoder_input, src_masks, tgt_masks
            )
            output_transpose = val_output.transpose(
                -1, -2
            )  # output needs to be N, C, other dimension for torch cross entropy
            loss = loss_function(output_transpose, decoder_desired_output_val.long())
            val_losses.append(loss.item())
            if validation_step % 20 == 0:
                print("Completed validation step", validation_step)
            validation_step += 1

        print("epoch", i, "took", time.time() - epoch_time_start)
        print("avg training loss:", sum(train_losses) / len(train_losses))
        avg_val_loss = sum(val_losses) / len(val_losses)
        print("avg validation loss:", avg_val_loss)
        expected_train_output = tgt_tokenizer.decode(
            decoder_desired_output_train[0].tolist()
        )
        print("expected train output", expected_train_output)
        decoded_output = decode(train_output[0], tgt_tokenizer)
        print("decoded train output:", decoded_output)
        expected_val_output = tgt_tokenizer.decode(
            decoder_desired_output_val[0].tolist()
        )
        print("expected validation output", expected_val_output)
        decoded_output = decode(val_output[0], tgt_tokenizer)
        print("decoded validation output:", decoded_output)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(encoder_decoder_transformer.state_dict(), state_dict_filename)
            print("Saved model state dict to", state_dict_filename)
            num_fails = 0
        else:
            print("Average validation loss did not decrease from ", best_val_loss)
            num_fails += 1
            print("Failed to decrease the average validation loss", num_fails, "times.")
            if num_fails >= 2:
                print("Stopping training")
                break
        print()
        print()
