
#!/bin/bash
### xvfb-run bash batch_develop_init_states.sh


# 1. 定义文件夹路径 (请根据你的实际路径进行调整)
# 假设你的 bddl 文件都存放在这个目录下
BDDL_DIR="/home/sylvia/sylvia/bigai/03vla/RoboCerebra/data/bddl_files/Random_Disturbance"
TARGET_DIR="/home/sylvia/sylvia/bigai/03vla/RoboCerebra/data/init_files/Random_Disturbance"


# 2. 确保目标文件夹存在
mkdir -p "$TARGET_DIR"

echo "开始批量生成初始状态..."
echo "目标目录: $TARGET_DIR"

# 3. 遍历目录下所有的 .bddl 文件
for bddl_file in "$BDDL_DIR"/*.bddl; do
    # 防止空文件夹导致星号被当成文件名
    if [ ! -f "$bddl_file" ]; then
        echo "❌ 错误: 在 $BDDL_DIR 中没有找到 .bddl 文件!"
        break
    fi

    filename=$(basename "$bddl_file")
    
    echo "=========================================================="
    echo "🚀 正在处理: $filename"
    
    # 4. 调用你的 Python 脚本
    python scripts/mem_1_develop_init_states.py \
        --bddl-file "$bddl_file" \
        --target-path "$TARGET_DIR"
        
    echo "✅ 完成处理: $filename"
done

echo "🎉 所有 BDDL 文件的初始状态生成完毕！"