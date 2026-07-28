import torch

from solo.methods.base import BaseMethod
from solo.utils.muon import Muon
from solo.utils.riemannian import RiemannianAdam


def test_muon_updates_matrix_and_auxiliary_parameters():
    matrix = torch.nn.Parameter(torch.randn(4, 3))
    vector = torch.nn.Parameter(torch.randn(3))
    optimizer = Muon(
        [
            {"params": [matrix], "use_muon": True, "lr": 0.02},
            {"params": [vector], "use_muon": False, "lr": 1e-3},
        ]
    )
    before_matrix, before_vector = matrix.detach().clone(), vector.detach().clone()
    (matrix.square().sum() + vector.square().sum()).backward()
    optimizer.step()
    assert not torch.equal(matrix, before_matrix)
    assert not torch.equal(vector, before_vector)


def test_riemannian_stiefel_retraction_for_rectangular_matrix():
    matrix = torch.nn.Parameter(torch.randn(5, 3))
    optimizer = RiemannianAdam(
        [{"params": [matrix], "use_riemannian": True, "lr": 1e-3}],
        manifold="stiefel",
    )
    (matrix.square().sum()).backward()
    optimizer.step()
    gram = matrix.transpose(0, 1) @ matrix
    assert torch.allclose(gram, torch.eye(3), atol=1e-5, rtol=1e-5)


def test_riemannian_row_stiefel_retraction():
    matrix = torch.nn.Parameter(torch.randn(3, 5))
    optimizer = RiemannianAdam(
        [{"params": [matrix], "use_riemannian": True, "lr": 1e-3}],
        manifold="stiefel",
    )
    (matrix.square().sum()).backward()
    optimizer.step()
    gram = matrix @ matrix.transpose(0, 1)
    assert torch.allclose(gram, torch.eye(3), atol=1e-5, rtol=1e-5)


def test_special_optimizers_only_mark_encoder_hidden_weights():
    model = object.__new__(BaseMethod)
    torch.nn.Module.__init__(model)
    model.extra_optimizer_args = {"aux_lr": 1e-3}
    model.backbone = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 8, kernel_size=3),
    )
    projector = torch.nn.Linear(8, 4)
    classifier = torch.nn.Linear(8, 2)
    groups = [
        {"name": "backbone", "params": model.backbone.parameters()},
        {"name": "projector", "params": projector.parameters()},
        {"name": "classifier", "params": classifier.parameters()},
    ]

    split = model._split_special_optimizer_groups(groups, "muon")
    muon_params = {
        id(parameter)
        for group in split
        if group["use_muon"]
        for parameter in group["params"]
    }

    assert id(model.backbone[2].weight) in muon_params
    assert id(model.backbone[0].weight) not in muon_params
    assert id(projector.weight) not in muon_params
    assert id(classifier.weight) not in muon_params
    first_conv_group = next(
        group
        for group in split
        if not group["use_muon"]
        and any(id(parameter) == id(model.backbone[0].weight) for parameter in group["params"])
    )
    assert first_conv_group["lr"] == 1e-3
