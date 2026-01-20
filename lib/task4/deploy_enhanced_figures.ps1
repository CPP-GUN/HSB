# Task4 图表快速部署脚本

# 安装必要的Python库
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "Task4 图表优化环境配置" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan

# 检查Python环境
Write-Host "`n[1/4] 检查Python环境..." -ForegroundColor Green
python --version

if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: Python未安装或未添加到PATH" -ForegroundColor Red
    exit 1
}

# 安装核心依赖
Write-Host "`n[2/4] 安装核心可视化库..." -ForegroundColor Green
$packages = @(
    "matplotlib>=3.5.0",
    "pandas>=1.3.0",
    "numpy>=1.21.0",
    "seaborn>=0.11.0",
    "scipy>=1.7.0"
)

foreach ($pkg in $packages) {
    Write-Host "  → 安装 $pkg" -ForegroundColor Cyan
    pip install $pkg --upgrade --quiet
}

# 安装增强库（可选）
Write-Host "`n[3/4] 安装增强功能库（可选）..." -ForegroundColor Green
$optional_packages = @(
    "plotly>=5.0.0",      # 桑基图
    "squarify>=0.4.3",    # 树状图
    "adjustText>=0.8",    # 标签防重叠
    "kaleido>=0.2.0"      # Plotly导出PDF
)

foreach ($pkg in $optional_packages) {
    Write-Host "  → 尝试安装 $pkg" -ForegroundColor Cyan
    pip install $pkg --upgrade --quiet 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    警告: $pkg 安装失败（非必需）" -ForegroundColor Yellow
    }
}

# 运行增强版脚本
Write-Host "`n[4/4] 生成增强版图表..." -ForegroundColor Green
Write-Host "  → 执行: plot_task4_figures_enhanced.py" -ForegroundColor Cyan

$script_path = Join-Path $PSScriptRoot "plot_task4_figures_enhanced.py"

if (Test-Path $script_path) {
    python $script_path
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n" -NoNewline
        Write-Host ("=" * 70) -ForegroundColor Green
        Write-Host "✅ 图表生成成功！" -ForegroundColor Green
        Write-Host ("=" * 70) -ForegroundColor Green
        
        Write-Host "`n输出目录:" -ForegroundColor Yellow
        $output_dir = Join-Path $PSScriptRoot "..\..\figure\task4"
        $abs_output = Resolve-Path $output_dir
        Write-Host "  $abs_output" -ForegroundColor Cyan
        
        Write-Host "`n生成的文件:" -ForegroundColor Yellow
        Get-ChildItem -Path $abs_output -Filter "*.pdf" | ForEach-Object {
            $size_mb = [math]::Round($_.Length / 1MB, 2)
            Write-Host "  ✓ $($_.Name) ($size_mb MB)" -ForegroundColor Cyan
        }
        
        # 检查HTML文件
        Get-ChildItem -Path $abs_output -Filter "*.html" | ForEach-Object {
            Write-Host "  ✓ $($_.Name) (交互式)" -ForegroundColor Magenta
        }
        
        Write-Host "`n对比文档:" -ForegroundColor Yellow
        $guide_path = Join-Path $PSScriptRoot "VISUALIZATION_OPTIMIZATION_GUIDE.md"
        Write-Host "  📄 $guide_path" -ForegroundColor Cyan
        
    } else {
        Write-Host "`n❌ 图表生成失败！" -ForegroundColor Red
        Write-Host "请检查错误信息并确保数据文件存在" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "`n❌ 错误: 找不到脚本文件" -ForegroundColor Red
    Write-Host "  预期路径: $script_path" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n" -NoNewline
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "🎉 部署完成！" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Cyan

Write-Host "`n📌 下一步操作:" -ForegroundColor Yellow
Write-Host "  1. 查看输出图表 PDF 文件" -ForegroundColor White
Write-Host "  2. 在浏览器打开 HTML 交互式图表（如有）" -ForegroundColor White
Write-Host "  3. 阅读 VISUALIZATION_OPTIMIZATION_GUIDE.md 了解优化细节" -ForegroundColor White
Write-Host "  4. 根据需要调整配色和样式参数`n" -ForegroundColor White
