import math
import numpy as np
import torch,torchvision

from torchofdreaveler import nn, optim, no_grad
from torchofdreaveler.training.checkpoint import save_checkpoint, load_checkpoint


def _acc_err(logits, yb):
    pred = logits.numpy().argmax(axis=1)
    err = np.mean(pred != yb)
    return float(err), 1.0 - float(err)


def _as_float(value):
    return float(value.numpy()) if hasattr(value, "numpy") else float(value)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_err = 0.0
    total = 0
    for xb, yb in loader:
        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = xb.shape[0]
        total_loss += _as_float(loss) * batch_size
        err, _ = _acc_err(logits, yb)
        total_err += err * batch_size
        total += batch_size

    if total == 0:
        return {"loss": 0.0, "err": 0.0, "acc": 0.0}
    avg_loss = total_loss / total
    avg_err = total_err / total
    return {"loss": avg_loss, "err": avg_err, "acc": 1.0 - avg_err}


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_err = 0.0
    total = 0
    with no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            batch_size = xb.shape[0]
            total_loss += _as_float(loss) * batch_size
            err, _ = _acc_err(logits, yb)
            total_err += err * batch_size
            total += batch_size

    if total == 0:
        return {"loss": 0.0, "err": 0.0, "acc": 0.0}
    avg_loss = total_loss / total
    avg_err = total_err / total
    return {"loss": avg_loss, "err": avg_err, "acc": 1.0 - avg_err}


def _get_writer(log_dir, writer):
    if writer is not None:
        return writer, False
    if log_dir is None:
        return None, False
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("TensorBoard is not available") from exc
    return SummaryWriter(log_dir=log_dir), True


def fit(model, train_loader, test_loader, optimizer, epochs=10, criterion=None, scheduler=None,
        checkpoint_path=None, save_every=1, resume_from=None, log_dir=None, writer=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    if resume_from is not None:
        last_epoch, _ = load_checkpoint(resume_from, model, optimizer)
        if last_epoch is not None:
            start_epoch = last_epoch + 1

    writer, should_close = _get_writer(log_dir, writer)

    print("| Epoch | Train Loss | Train Acc | Train Err | Test Loss | Test Acc | Test Err |")
    try:
        for epoch in range(start_epoch, epochs):
            train_stats = train_one_epoch(model, train_loader, criterion, optimizer)
            test_stats = evaluate(model, test_loader, criterion)
            print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |   {:.5f} |   {:.5f} |   {:.5f} |".format(
                epoch,
                train_stats["loss"],
                train_stats["acc"],
                train_stats["err"],
                test_stats["loss"],
                test_stats["acc"],
                test_stats["err"],
            ))

            if writer is not None:
                writer.add_scalar("loss/train", train_stats["loss"], epoch)
                writer.add_scalar("loss/test", test_stats["loss"], epoch)
                writer.add_scalar("acc/train", train_stats["acc"], epoch)
                writer.add_scalar("acc/test", test_stats["acc"], epoch)
                writer.add_scalar("err/train", train_stats["err"], epoch)
                writer.add_scalar("err/test", test_stats["err"], epoch)
                writer.add_scalar("lr", optimizer.lr, epoch)

            if scheduler is not None:
                scheduler.step()

            if checkpoint_path and save_every and (epoch + 1) % save_every == 0:
                save_checkpoint(checkpoint_path, model, optimizer=optimizer, epoch=epoch)
    finally:
        if should_close and writer is not None:
            writer.close()


def train(model, train_loader, test_loader, epochs=10, lr=1e-3, using_adam=False,
          step_size=None, gamma=0.1, momentum=0.0, use_cos_lr=False, min_lr=0.0,
          checkpoint_path=None, save_every=1, resume_from=None, log_dir=None, writer=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) if using_adam else optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
    )

    scheduler = None
    if use_cos_lr:
        scheduler = optim.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=min_lr)
    elif step_size is not None:
        scheduler = optim.StepLR(optimizer, step_size=step_size, gamma=gamma)

    fit(
        model,
        train_loader,
        test_loader,
        optimizer,
        epochs=epochs,
        criterion=criterion,
        scheduler=scheduler,
        checkpoint_path=checkpoint_path,
        save_every=save_every,
        resume_from=resume_from,
        log_dir=log_dir,
        writer=writer,
    )
