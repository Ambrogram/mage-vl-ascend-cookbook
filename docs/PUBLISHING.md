# GitHub 发布

仓库根可直接上传，或自行git init并提交；不要把父级GPTQModel仓库、历史outputs、私人BML连接器一起上传。作者署名可在发布前填入；本包未代建GitHub仓库或推送。

运行 `python tools/check_release.py` 与 `python -m unittest discover -s tests -v`。
`python tools/package_release.py --out /your/new/mage-vl-ascend-cookbook.zip` 生成SHA清单和ZIP。
脚本拒绝覆盖既有ZIP，只按仓库文件白名单打包，不包含.git、缓存和实机runs。

上传前逐项阅读LICENSE、NOTICE、SOURCES；模型许可与仓库代码许可独立。Mamba的Apache-2.0文件和许可必须保留。用户自行取得模型与样本；不上传任何pickle权重、照片、测试录像、wheel或平台账户。

历史证据只发字段白名单摘要及原始SHA，不能假装原始张量日志已全部公开。发布检查发现内部路径、平台地址、疑似密钥、二进制或坏链接则停止；修正后再打包，不静默删除失败记录。新硬件补测在独立runs目录完成，审查后只加入脱敏摘要。

推荐GitHub描述：Evidence-led Mage-VL deployment on Ascend 910B2: native BF16 inference, FP32 StreamMind gate adaptation, bounded video replay and reproducible deployment checklists.
推荐标签：ascend、910b、mage-vl、torch-npu、multimodal、deployment、streammind。
