

# GPU, batch size, epoch cố định
# GPU=2
# BATCH=4
# EPOCH=100

# # 5 learning rate khác nhau
# for LR in 5e-4 3e-4 5e-3 1e-4 2e-4
# do
#     for st in 0 
#     do
#         echo "🔹 Running with lr=$LR, stu-type=$st ..."
#         python main_overlap_tag2.py \
#             --gpu $GPU \
#             --batch-size $BATCH \
#             --num-epochs $EPOCH \
#             --lr $LR \
#             --stu-type $st
#     done
# done
# CUDA_VISIBLE_DEVICES=1 python main_overlap_tag.py --gpu 0 --batch-size 4 --num-epochs 100 --lr 1e-3 --stu-type 0  \
# & CUDA_VISIBLE1DEVICES=0 python main_overlap_tag.py --gpu 0 --batch-size 4 --num-epochs 100 --lr 1e-2 --stu-type 0 \
# & CUDA_VISIBLE_DEVICES=2 python main_overlap_tag.py --gpu 0 --batch-size 64 --num-epochs 100 --lr 1e-2 --stu-type 0
# CUDA_VISIBLE_DEVICES=0 python main_overlap_tag.py --gpu 0 --batch-size 4 --num-epochs 100 --lr 5e-4 --stu-type 0
#  CUDA_VISIBLE_DEVICES=1 python main_overlap_tag.py --gpu 0 --batch-size 4 --num-epochs 100 --lr 1e-2 --stu-type 0


CUDA_VISIBLE_DEVICES=0 python main_overlap_tag.py --gpu 0 --batch-size 64 --num-epochs 200 --lr 3e-4 --stu-type 0 --distill_type 2 &

# Đợi 1 phút (60 giây) rồi chạy job 2
sleep 60
CUDA_VISIBLE_DEVICES=1 python main_overlap_tag.py --gpu 0 --batch-size 64 --num-epochs 200 --lr 3e-4 --stu-type 0 --distill_type 21 &

# Đợi thêm 1 phút nữa rồi chạy job 3
sleep 60
CUDA_VISIBLE_DEVICES=2 python main_overlap_tag.py --gpu 0 --batch-size 64 --num-epochs 200 --lr 3e-4 --stu-type 0 --distill_type 22 &