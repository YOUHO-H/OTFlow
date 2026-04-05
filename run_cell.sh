CUDA_VISIBLE_DEVICES=0 python main_cell.py \
                --outdir ./results/4i/drug-cisplatin/model-cellot \
                --config ./configs/tasks/4i.yaml \
                --config ./configs/models/cellot.yaml \
                --config.data.target cisplatin