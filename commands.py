"""
Generate commands.txt for running experiments, where each line is a single python run command
Then commands.txt is used by run.sh to run experiments in series.
"""

datasets = ["coco_val", "cityscapes", "bdd100k_train"]

risk_control = {
    "std_conf": ["std_rank"],
    "ens_conf": ["ens_rank"],
    "cqr_conf": ["cqr_rank"],
}

# with conformal baselines
# risk_control = {
#     "std_conf": ["std_rank", "std_bonf"],
#     "ens_conf": ["ens_rank", "ens_bonf", "base_ens"],
#     "cqr_conf": ["cqr_rank", "cqr_bonf"],
#     "base_conf": ["base_rank", "base_max", "base_bonf"],
# }

model_id = "x101fpn"
label_control = {"top_singleton": "top", "oracle": "oracle", "full": "full"}
def_label_control = "class_threshold"


def main():
    file = "commands.txt"
    print(f"Writing commands to {file}")

    with open(file, "w") as f:
        for d in datasets:
            for rc in risk_control.keys():
                for cfg in risk_control[rc]:
                    s = f"python conformalbb/main.py --config_file=cfg_{cfg} --config_path=conformalbb/config/{d} --run_collect_pred --save_file_pred --risk_control={rc} --alpha=0.1 --label_set=class_threshold --label_alpha=0.01 --run_risk_control --save_file_control --save_label_set --run_eval --save_file_eval --file_name_suffix=_{cfg}_class --device=cuda"
                    f.write(s + "\n")
                    s = f"python conformalbb/main.py --config_file=cfg_{cfg} --config_path=conformalbb/config/{d} --load_collect_pred={rc}_{model_id}_{cfg}_class --risk_control={rc} --alpha=0.1 --label_set=oracle --label_alpha=0.01 --run_risk_control --save_file_control --save_label_set --run_eval --save_file_eval --file_name_suffix=_{cfg}_oracle --device=cuda"
                    f.write(s + "\n")
                    s = f"python conformalbb/main.py --config_file=cfg_{cfg} --config_path=conformalbb/config/{d} --load_collect_pred={rc}_{model_id}_{cfg}_class --risk_control={rc} --alpha=0.1 --label_set=top_singleton --label_alpha=0.01 --run_risk_control --save_file_control --run_eval --save_file_eval --file_name_suffix=_{cfg}_top --device=cuda"
                    f.write(s + "\n")
                    s = f"python conformalbb/main.py --config_file=cfg_{cfg} --config_path=conformalbb/config/{d} --load_collect_pred={rc}_{model_id}_{cfg}_class --risk_control={rc} --alpha=0.1 --label_set=full --label_alpha=0.01 --run_risk_control --save_file_control --run_eval --save_file_eval --file_name_suffix=_{cfg}_full --device=cuda"
                    f.write(s + "\n")


if __name__ == "__main__":
    main()
