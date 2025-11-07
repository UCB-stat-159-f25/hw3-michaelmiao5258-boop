.PHONY: env html clean

# 安装/更新本作业需要的包
env:
	python -m pip install -e .
	python -m pip install -U mystmd pytest

# 构建 MyST 静态站点到 _build/html/
html:
	myst build --html

# 清理构建产物和缓存
clean:
	rm -rf _build
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
