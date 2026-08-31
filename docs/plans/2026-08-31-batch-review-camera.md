# 批量审核与摄像头调校实现计划

> **注意：** 使用 executing-plans skill 逐任务实现此计划。

**目标：** 实现持久化批量审核、恢复与真假处理解耦、桌面切换兜底，以及可安全降级的摄像头对焦/曝光设置。

**架构：** 在 `main.py` 增加独立 `ReviewQueue` 管理图片和 JSON 元数据，状态机只负责入队和恢复；`ui.py` 增加审核对话框与摄像头调校控件。桌面切换和摄像头属性封装成小函数，便于无副作用测试。

**技术栈：** Python、OpenCV、NumPy、PyQt6、unittest、PyInstaller、Win32 API

---

### 任务 1：持久化审核队列

**文件：**
- 修改：`main.py`（替换 `FaceReviewSession`）
- 创建：`test_review_queue.py`

**步骤 1：编写失败测试**

覆盖连续添加两次事件、只保留事件最高分图片、重新实例化后仍能列出记录、删除记录同时删除图片和 JSON、损坏 JSON 被隔离。

**步骤 2：运行测试验证失败**

运行：`python -m unittest test_review_queue.py -v`

预期：因 `ReviewQueue` 尚不存在而 FAIL。

**步骤 3：编写最小实现**

实现 `ReviewQueue.start_event()`、`consider()`、`finish_event()`、`list_items()` 和 `delete()`；采用 `review_queue/<id>.jpg` 与 `<id>.json`。

**步骤 4：运行测试验证通过**

运行：`python -m unittest test_review_queue.py -v`

预期：全部 PASS。

**步骤 5：提交**

```bash
git add main.py test_review_queue.py
git commit -m "功能: 添加持久化识别审核队列"
```

### 任务 2：状态机解耦恢复与审核

**文件：**
- 修改：`main.py`（`run_loop`）
- 修改：`ui.py`（`Engine` 与主窗口按钮）
- 修改：`test_face_learning.py`
- 修改：`test_review_controls.py`

**步骤 1：编写失败测试**

验证每次 `DEFEND` 都入队、恢复事件不学习也不删除、下一次防御形成第二条记录，主界面恢复为单独恢复按钮并显示审核数量。

**步骤 2：运行测试验证失败**

运行：`python -m unittest test_face_learning.py test_review_controls.py -v`

预期：旧的即时确认接口与新期望不一致而 FAIL。

**步骤 3：编写最小实现**

删除运行循环的确认/误报事件，防御事件结束后保留队列记录；恢复只切回界面并回到巡逻态。主界面增加恢复按钮与审核入口。

**步骤 4：运行测试验证通过**

运行同上，预期全部 PASS。

**步骤 5：提交**

```bash
git add main.py ui.py test_face_learning.py test_review_controls.py
git commit -m "功能: 检测记录与恢复操作解耦"
```

### 任务 3：批量审核服务与界面

**文件：**
- 修改：`main.py`（批量学习结果）
- 修改：`ui.py`（`ReviewDialog`）
- 创建：`test_batch_review.py`
- 修改：`test_review_controls.py`

**步骤 1：编写失败测试**

测试多选确认时成功/重复/无效分别计数，多选误报删除，未选择记录保留；Qt 测试验证审核按钮与记录数量。

**步骤 2：运行测试验证失败**

运行：`python -m unittest test_batch_review.py test_review_controls.py -v`

预期：批量审核 API 和对话框尚不存在而 FAIL。

**步骤 3：编写最小实现**

实现 `ReviewQueue.confirm_many()`、`reject_many()` 与缩略图多选对话框；完成后刷新数量和日志。

**步骤 4：运行测试验证通过**

运行同上，预期全部 PASS。

**步骤 5：提交**

```bash
git add main.py ui.py test_batch_review.py test_review_controls.py
git commit -m "功能: 添加批量识别审核界面"
```

### 任务 4：桌面切换兜底

**文件：**
- 修改：`main.py`（Win+D 封装与状态机）
- 创建：`test_desktop_fallback.py`

**步骤 1：编写失败测试**

验证无护身窗口时调用 `toggle_desktop()`，恢复时再次调用；有护身窗口时不调用桌面兜底。

**步骤 2：运行测试验证失败**

运行：`python -m unittest test_desktop_fallback.py -v`

预期：缺少桌面兜底函数或未接入状态机而 FAIL。

**步骤 3：编写最小实现**

封装 Win+D 键盘事件，记录当前防御是否使用桌面兜底；恢复及退出时成对切换。

**步骤 4：运行测试验证通过**

运行同上，预期全部 PASS。

**步骤 5：提交**

```bash
git add main.py test_desktop_fallback.py
git commit -m "功能: 未配置护身窗口时切换桌面"
```

### 任务 5：摄像头对焦与曝光调校

**文件：**
- 修改：`main.py`（摄像头属性与检测增强）
- 修改：`ui.py`（设置控件）
- 修改：`config.example.json`
- 创建：`test_camera_controls.py`

**步骤 1：编写失败测试**

使用伪采集设备验证属性映射、实际值回读、不支持属性时返回提示、检测增强不修改原帧。

**步骤 2：运行测试验证失败**

运行：`python -m unittest test_camera_controls.py -v`

预期：摄像头控制函数尚不存在而 FAIL。

**步骤 3：编写最小实现**

实现 `apply_camera_controls()` 和 `prepare_detection_frame()`；设置对话框加入自动对焦、焦距、自动曝光、曝光、逆光补偿和亮度均衡选项。

**步骤 4：运行测试验证通过**

运行同上，预期全部 PASS。

**步骤 5：提交**

```bash
git add main.py ui.py config.example.json test_camera_controls.py
git commit -m "功能: 添加摄像头对焦曝光调校"
```

### 任务 6：文档、全量验证与发布

**文件：**
- 修改：`README.md`
- 修改：`laocao/README.md`
- 修改：`LaoCaoMirror.spec`（仅在新增依赖需要时）

**步骤 1：更新说明**

记录新审核流程、桌面兜底、摄像头调校、数据目录和低配建议。

**步骤 2：运行全量安全测试**

运行：

```bash
python -m py_compile main.py ui.py
python -m unittest test_detection_pipeline.py test_face_alignment.py test_face_learning.py test_review_queue.py test_batch_review.py test_desktop_fallback.py test_camera_controls.py test_review_controls.py -v
```

预期：全部 PASS，且不打开真实摄像头或切换桌面。

**步骤 3：构建与 EXE 冷启动验证**

复用已验证 CPU 运行库构建新 EXE，使用图片源和截图模式验证 Qt/DLL、队列入口和防御状态。

**步骤 4：提交与发布**

提交文档，推送 `main`，创建下一补丁版本标签和 GitHub Release，上传 CPU 压缩包并核对 `uploaded` 状态与 SHA256。

