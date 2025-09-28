# 本地保存路径
LOCAL_DIR="./MindSpeed-RL/json_file"

# 远程服务器列表（格式：user@ip）
SERVERS=(
    "root@IP"
    "root@IP"
)

# 远程 json_file 文件夹路径
REMOTE_DIR="./MindSpeed-RL/json_file"

# 遍历每台服务器并同步文件
for SERVER in "${SERVERS[@]}"; do
    scp "$SERVER:$REMOTE_DIR/*.json" "$LOCAL_DIR/"
    if [ $? -ne 0 ]; then
        echo " 从 $SERVER 同步失败，请检查 SSH 连接或路径是否正确"
    else
        echo " $SERVER 同步完成"
    fi
done

echo " 所有文件已同步到 $LOCAL_DIR"
echo " 当前目录中 JSON 文件数量：$(ls -1 "$LOCAL_DIR"/*.json 2>/dev/null | wc -l)"