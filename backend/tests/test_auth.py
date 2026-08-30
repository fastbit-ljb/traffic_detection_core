import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app import main
from app.core.security import get_password_hash, verify_token
from app.services.project_service import ProjectRepository


def test_history_is_isolated_by_user(tmp_path):
    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()
    first = repository.create_user("alice", get_password_hash("password123"))
    second = repository.create_user("bob", get_password_hash("password123"))
    assert first and second

    alice_entry = repository.add_history("image", "alice.jpg", {}, 1, 0.1, None, "base", user_id=first["id"])
    bob_entry = repository.add_history("image", "bob.jpg", {}, 2, 0.1, None, "base", user_id=second["id"])

    assert [entry["id"] for entry in repository.list_history(user_id=first["id"])] == [alice_entry["id"]]
    assert [entry["id"] for entry in repository.list_history(user_id=second["id"])] == [bob_entry["id"]]
    assert repository.delete_history(bob_entry["id"], first["id"]) is None
    assert repository.list_history(user_id=second["id"])[0]["id"] == bob_entry["id"]


@pytest.mark.asyncio
async def test_background_jobs_are_isolated_by_user(tmp_path):
    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()
    first = repository.create_user("alice", get_password_hash("password123"))
    second = repository.create_user("bob", get_password_hash("password123"))
    assert first and second

    alice_job = repository.create_job("video", {"source_name": "alice.mp4"}, user_id=first["id"])
    bob_job = repository.create_job("training", {"dataset_id": "bob-dataset"}, user_id=second["id"])

    assert [job["id"] for job in repository.list_jobs(user_id=first["id"])] == [alice_job["id"]]
    assert repository.get_job(alice_job["id"], second["id"]) is None
    assert repository.get_job(alice_job["id"], first["id"])["id"] == alice_job["id"]

    with pytest.raises(HTTPException) as error:
        await main.get_video_job(alice_job["id"], repository, second)
    assert error.value.status_code == 404

    visible = await main.get_video_job(alice_job["id"], repository, first)
    assert visible["id"] == alice_job["id"]
    assert repository.get_job(bob_job["id"], first["id"]) is None


@pytest.mark.asyncio
async def test_register_and_login_return_bearer_token(tmp_path):
    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()

    registered = await main.register_account(main.AuthCredentials(username="alice", password="password123"), repository)
    assert registered["user"]["username"] == "alice"
    assert verify_token(registered["access_token"])["sub"] == registered["user"]["id"]

    logged_in = await main.login_account(main.AuthCredentials(username="alice", password="password123"), repository)
    assert logged_in["user"]["id"] == registered["user"]["id"]

    with pytest.raises(HTTPException) as error:
        await main.login_account(main.AuthCredentials(username="alice", password="wrongpass"), repository)
    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_current_user_dependency_rejects_missing_or_invalid_token(tmp_path):
    repository = ProjectRepository(tmp_path / "storage")
    repository.initialize()

    with pytest.raises(HTTPException) as missing:
        await main.get_current_user(None, repository)
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as invalid:
        await main.get_current_user(HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid"), repository)
    assert invalid.value.status_code == 401
