"""CPU-only reset/step checks; these are not policy evaluations."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def alfworld_check():
    from agent_system.environments.env_package.alfworld.envs import AlfworldWorker, load_config_file
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

    config = load_config_file(ROOT / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml")
    base = get_environment("AlfredTWEnv")(config, train_eval="eval_in_distribution")
    if len(base.game_files) != 140:
        raise ValueError(f"Expected 140 valid_seen gamefiles, got {len(base.game_files)}")
    worker = AlfworldWorker(config, 1000, base, game_index=0)
    try:
        first, info = worker.reset(0)
        filename = info["extra.gamefile"][0]
        obs, _, _, _ = worker.step("look")
        second, info2 = worker.reset(0)
        assert info2["extra.gamefile"][0] == filename and first == second
        assert obs[0]
        return {"dataset": "alfworld", "valid_seen_games": 140, "fixed_reset": True, "step": "look", "passed": True}
    finally:
        worker.env.close()


def webshop_check():
    from agent_system.environments.env_package.webshop.envs import WebshopWorker

    path = ROOT / "agent_system/environments/env_package/webshop/webshop/data"
    worker = WebshopWorker([1000], {"observation_mode": "text", "num_products": None, "human_goals": False,
                                   "file_path": str(path / "items_shuffle_1000.json"),
                                   "attr_path": str(path / "items_ins_v2_1000.json")})
    try:
        first = worker.reset([0])[0]
        second = worker.reset([0])[0]
        assert first[0] == second[0] and first[1]["goal_index"] == 0
        result = worker.step(["search[shirt]"])[0]
        assert result[0] and "task_score" in result[3]
        goals = worker.get_goals()
        assert len(goals) >= 500
        return {"dataset": "webshop", "available_goals": len(goals), "fixed_test_goals": 500,
                "fixed_reset": True, "step": "search[shirt]", "passed": True}
    finally:
        worker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["alfworld", "webshop"])
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' for this CPU-only check")
    result = alfworld_check() if args.dataset == "alfworld" else webshop_check()
    print(json.dumps(result, indent=2))
