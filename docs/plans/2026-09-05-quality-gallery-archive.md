# 高质量训练图库与照片归档实现计划

> **注意：** 使用 executing-plans skill 逐任务实现此计划。

**目标：** 超过自动学习上限时保留质量最高的照片，并将落选照片按日期归档而非删除。

**架构：** `FaceRecognizer` 负责视觉/身份综合评分和活跃样本淘汰；归档目录位于图库子目录，不被图库加载器扫描。`ReviewQueue` 和审核界面区分加入训练、直接归档与替换归档三种结果。

**技术栈：** Python、OpenCV、NumPy、unittest、PyQt6。

---

### 任务 1：用失败测试定义质量淘汰和归档

**文件：**
- 修改: `test_face_learning.py`

**步骤 1：编写失败测试**

将旧的“删除最旧照片”测试替换为：构造一张旧但清晰且身份一致的照片、一张较新但模糊/身份偏离的照片，再学习一张高质量照片；断言低质量照片进入 `archive/日期/`，旧的高质量照片仍在活跃图库。

再增加新候选本身质量最低的测试，断言返回 `status == "archived"`，活跃自动样本数量不变，候选文件只存在于归档目录。

**步骤 2：运行测试验证失败**

运行：

```powershell
python -m unittest test_face_learning.FaceGalleryLearningTests -v
```

预期：旧实现仍删除最旧文件，归档断言失败。

### 任务 2：实现评分与无覆盖归档

**文件：**
- 修改: `main.py`
- 测试: `test_face_learning.py`

**步骤 1：实现最小评分函数**

新增视觉评分：拉普拉斯方差经对数归一化为清晰度，平均亮度和极端像素比例组成曝光分。新增身份锚点评分，手工原图优先；最终按 `0.70 * 身份 + 0.20 * 清晰度 + 0.10 * 曝光` 排序。

**步骤 2：实现归档函数**

从 `learned_YYYYMMDD_...` 解析日期，创建 `gallery/archive/YYYY-MM-DD/`，使用无覆盖文件名移动图片。移动成功后才删除对应内存 embedding。

**步骤 3：替换淘汰循环**

每次学习后给活跃 `learned_` 文件评分，超过上限时反复归档最低分。返回 `archived` 路径列表；若新文件被归档，返回 `status: archived`。

**步骤 4：运行测试验证通过**

```powershell
python -m unittest test_face_learning.FaceGalleryLearningTests -v
```

预期：全部通过。

### 任务 3：批量审核结果与界面提示

**文件：**
- 修改: `main.py`
- 修改: `ui.py`
- 修改: `test_batch_review.py`

**步骤 1：添加失败测试**

让测试识别器分别返回 `learned`、`archived`、`duplicate`，断言队列移除三者、保留无效记录，并统计 `archived` 与被替换归档数量。

**步骤 2：实现统计和文案**

扩展 `confirm_many` 结果字典；审核窗口显示“加入训练、质量不足归档、替换归档、重复、无法对齐、失败”。

**步骤 3：验证**

```powershell
python -m unittest test_batch_review test_review_controls -v
```

预期：全部通过。

### 任务 4：文档、完整回归与提交

**文件：**
- 修改: `README.md`

**步骤 1：更新文档**

说明训练集只保留质量最高的 30 张，旧/差照片进入归档目录且不参与识别。

**步骤 2：完整验证**

```powershell
python -m unittest test_batch_review test_camera_controls test_desktop_fallback test_detection_pipeline test_engine_lifecycle test_face_alignment test_face_learning test_inference_backend test_review_controls test_review_queue
python test_roi_dialog.py
python -m py_compile main.py ui.py ui_cpu.py ui_gpu.py
git diff --check
```

预期：所有测试通过，无语法或空白错误。

**步骤 3：提交**

```powershell
git add main.py ui.py test_face_learning.py test_batch_review.py README.md docs/plans
git commit -m "功能: 按质量保留训练照片并归档落选样本"
```
