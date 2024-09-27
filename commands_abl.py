"""
Generate commands_abl.txt for running experiments, where each line is a single python run command
Then commands_abl.txt is used by run.sh to run experiments in series.
This script is used to generate commands for the ablations on different coverage levels.
"""

import os

datasets = ["coco_val", "cityscapes", "bdd100k_train"]

risk_control = {"std_conf": "std_rank", "ens_conf": "ens_rank", "cqr_conf": "cqr_rank"}

model_id = "x101fpn"

box_cov = {"85": 0.15, "90": 0.1, "95": 0.05}
label_cov = {"80": 0.2, "90": 0.1, "99": 0.01, "100": 0.001}


def main():
    file = "commands_abl.txt"
    print(f"Writing commands to {file}")

    with open(file, "w") as f:
        for d in datasets:
            for rc, cfg in risk_control.items():
                for bcov in box_cov.keys():
                    for lcov in label_cov.keys():
                        # this is the original setting, skip
                        if bcov == "90" and lcov == "99":
                            continue

                        if d == "bdd100k_train":
                            # don't save label set due to file sizes
                            s = f"python conformalbb/main.py --config_file=cfg_{cfg} --config_path=conformalbb/config/{d} --load_collect_pred={rc}_{model_id}_{cfg}_class --risk_control={rc} --alpha={box_cov[bcov]} --label_set=class_threshold --label_alpha={label_cov[lcov]} --run_risk_control --save_file_control --run_eval --save_file_eval --file_name_suffix=_{cfg}_class_{bcov}_{lcov} --device=cuda"
                            f.write(s + "\n")
                            s = f"python conformalbb/main.py --config_file=cfg_{cfg} --config_path=conformalbb/config/{d} --load_collect_pred={rc}_{model_id}_{cfg}_class --risk_control={rc} --alpha={box_cov[bcov]} --label_set=oracle --label_alpha={label_cov[lcov]} --run_risk_control --save_file_control --run_eval --save_file_eval --file_name_suffix=_{cfg}_oracle_{bcov}_{lcov} --device=cuda"
                            f.write(s + "\n")
                            s = f"python conformalbb/main.py --config_file=cfg_{cfg} --config_path=conformalbb/config/{d} --load_collect_pred={rc}_{model_id}_{cfg}_class --risk_control={rc} --alpha={box_cov[bcov]} --label_set=top_singleton --label_alpha={label_cov[lcov]} --run_risk_control --save_file_control --run_eval --save_file_eval --file_name_suffix=_{cfg}_top_{bcov}_{lcov} --device=cuda"
                            f.write(s + "\n")
                            s = f"python conformalbb/main.py --config_file=cfg_{cfg} --config_path=conformalbb/config/{d} --load_collect_pred={rc}_{model_id}_{cfg}_class --risk_control={rc} --alpha={box_cov[bcov]} --label_set=full --label_alpha={label_cov[lcov]} --run_risk_control --save_file_control --run_eval --save_file_eval --file_name_suffix=_{cfg}_full_{bcov}_{lcov} --device=cuda"
                            f.write(s + "\n")
                        else:
                            s = f"python conformalbb/main.py --config_file=cfg_{cfg} --config_path=conformalbb/config/{d} --load_collect_pred={rc}_{model_id}_{cfg}_class --risk_control={rc} --alpha={box_cov[bcov]} --label_set=class_threshold --label_alpha={label_cov[lcov]} --run_risk_control --save_file_control --save_label_set --run_eval --save_file_eval --file_name_suffix=_{cfg}_class_{bcov}_{lcov} --device=cuda"
                            f.write(s + "\n")
                            s = f"python conformalbb/main.py --config_file=cfg_{cfg} --config_path=conformalbb/config/{d} --load_collect_pred={rc}_{model_id}_{cfg}_class --risk_control={rc} --alpha={box_cov[bcov]} --label_set=oracle --label_alpha={label_cov[lcov]} --run_risk_control --save_file_control --save_label_set --run_eval --save_file_eval --file_name_suffix=_{cfg}_oracle_{bcov}_{lcov} --device=cuda"
                            f.write(s + "\n")
                            s = f"python conformalbb/main.py --config_file=cfg_{cfg} --config_path=conformalbb/config/{d} --load_collect_pred={rc}_{model_id}_{cfg}_class --risk_control={rc} --alpha={box_cov[bcov]} --label_set=top_singleton --label_alpha={label_cov[lcov]} --run_risk_control --save_file_control --run_eval --save_file_eval --file_name_suffix=_{cfg}_top_{bcov}_{lcov} --device=cuda"
                            f.write(s + "\n")
                            s = f"python conformalbb/main.py --config_file=cfg_{cfg} --config_path=conformalbb/config/{d} --load_collect_pred={rc}_{model_id}_{cfg}_class --risk_control={rc} --alpha={box_cov[bcov]} --label_set=full --label_alpha={label_cov[lcov]} --run_risk_control --save_file_control --run_eval --save_file_eval --file_name_suffix=_{cfg}_full_{bcov}_{lcov} --device=cuda"
                            f.write(s + "\n")


if __name__ == "__main__":
    main()
