# 掌纹生成器交接

日期：2026-10-20，2026-10-21 补了主线模板，之后按文献校正了手的比例、汗孔密度和纹型分布，并加了人群预设（`-population`）和随机性别。接手对象是 `palmprint_generator/`。原仓库的指纹生成器和汗孔/划痕程序没有改。

注意：2026-10-21 起，同一种子生成的整幅图与此前版本不同。原因有两个：主线不再消耗 `rng_`；`rng_` 的种子也改成先打散（见下）。方向场扰动、脊距噪声、初始噪声、手形抖动和纹型都跟着变了。之后加手形、解剖比例校正、人群预设和性别时又各变了一次。按种子号存过的旧数据不能再复现，以当天最后的版本为准。默认预设是 `han`（这是交接时的选择，改 `population.h` 里 `kPopulations` 的第一项即可换默认）。各随机数流的种子里叠加了预设名的哈希，同一种子在不同预设下是不同的手；测试里 50 个种子在 `han` 和 `japanese` 下纹型全部不同。

五条随机数流各自加盐打散，统一经 `stream_seed(k) = mix_seed(seed + k·2^32 + FNV1a(预设名))`：主线 `lines` k=0，指尖 `tips` k=1，`rng_` k=2，手形和性别 `shape` k=3，掌部指间区纹型 `palm` k=4。不打散时 `cv::RNG` 对相邻种子给出相近的前几个数：拇指纹型在种子 1–8 全是斗、9–13 全是箕。打散后 40 个种子里拇指斗占 23 个（设定 60%），最长同型连串 5。新加随机数流时也要加盐，不要和已有的共用同一个打散值，否则两条流完全相同。

## 现在能做什么

`palmprint.exe` 从种子生成整只手的掌面印：五指三节、指根三角、大鱼际、小鱼际、掌心、腕部，加上关节横纹和按解剖模板抽取的掌部主线。脊线按方向场和脊距迭代 Gabor 滤波长出来，连续，带分叉和断头。

已验证：

| 分辨率 | 种子 | 耗时 | 结果 |
| --- | --- | --- | --- |
| 250 dpi（默认） | 1，左手，`han` | 约 1.5 s | `deployment/output/palm_left.jpg`，1856×1962（女，手长 16.3 cm；画布随种子的手形变化） |
| 250 dpi | 1，`-right` | 约 1.5 s | `deployment/output/palm_right.jpg` |
| 500 dpi | 3 | 约 12–15 s | 脊线按真实比例，末节汗孔女约 53、男约 46 个/9 mm² |

放大看过的部位：指节横纹、指根三角汇流、大鱼际、小鱼际、主线边缘、各类末节纹型。250 dpi 和 500 dpi 都看过。

## 测试

```powershell
python palmprint_generator\tests\run_tests.py
```

需要 numpy 和 Pillow，约 5 分钟，全部通过时退出码为 0，输出在 `tests/_out/`（已忽略）。它会先跑一次 `build.bat`，然后检查：

- 命令行：`-h`、缺参数、未知参数、dpi < 40、无法识别的扩展名、非法种子（`-1`、`abc`、`12x`、`4294967296`、空串）、未知人群都以退出码 1 拒绝。
- 同一种子输出逐像素一致；`-right` 逐像素等于左手的水平翻转。
- `-layers` 的 6 个文件齐全、尺寸一致；背景纯白，褶皱只在手内。
- `generic` 预设 300 个种子（100 dpi）：全部正常退出、没有主线起笔警告；按指检查纹型（小指箕 > 80%、环指斗 > 35%、拇指斗 30–50%、桡侧箕和弓集中在示指、全部手指箕 55–70%）；I4 箕 30–65%；男 40–60%；男女平均手长、掌宽在文献区间内、男宽于女；平均 atd 33–48°。并打印各模板、纹型、掌部纹型的计数。
- `han`、`european`、`japanese` 各 200 个种子（60 dpi）：每指斗比例与预设表相差 < 11 个百分点，I4 箕相差 < 11 个百分点，男女平均手长相差 < 0.5 cm。
- 500 dpi 末节汗孔密度：`han` 种子 1（女）46–58、种子 4（男）40–51 个/9 mm²。
- 50 个种子在 `han` 和 `japanese` 下纹型不同的比例 > 40%（实测 50/50）。
- 12 个种子的手掌面积变异系数 > 5%。
- 种子 0、2147483647、4294967295 都能跑，后两个结果不同。
- dpi 40、100、250、500 都能跑，并打印耗时。

2026-10-21 最后一次跑的结果是 65/65 通过（含各 dpi 下“低 dpi 汗孔提示”是否按预期出现）。早先测试中修掉两个 bug：
- 扩展名无法识别（如 `.xyz`）时，`cv::imwrite` 抛异常导致程序崩溃，而且是在整套生成跑完后才崩。现在 `main` 一开始就用 `cv::haveImageWriter` 检查，生成过程也包了 `try/catch`。
- `-seed` 原来用 `atoi` 解析，超过 2147483647 的种子全被截成同一个，`-1` 则绕回 4294967295。现在用 `strtoull` 严格解析 0..4294967295。

没测的：dpi 没有上限，1000 dpi 以上的内存和耗时没试过。

## 仓库里的三块

- `gpu_fingerprint_generator/`：盛林翔 2022 年的 SFinGe 风格指纹生成器，CPU（MinGW）和 GPU（CUDA）两套。末节指纹的奇点公式从这里来。
- `add_pore_scratch/`：给已有指纹加汗孔和划痕。掌纹生成器没有调用它。
- `palmprint_generator/`：这次新增的整手掌纹。只依赖 OpenCV 4.5.4，不用 CUDA。

## 编译和运行

不需要 CUDA，也不需要把 OpenCV 装进系统。要求：Visual Studio（带 x64 C++ 工具，2017 及以上），能上网或手头有 OpenCV 4.5.4 的 Windows 安装包。

第一次在一台机器上编译，先取 OpenCV：

```powershell
powershell -ExecutionPolicy Bypass -File palmprint_generator\fetch_opencv.ps1
```

它下载 `7zr.exe` 和官方 `opencv-4.5.4-vc14_vc15.exe`（约 220 MB，缓存在 `third_party/_download/`），只解出头文件、`x64\vc15\lib\opencv_world454.lib` 和 `x64\vc15\bin\opencv_world454.dll`，放到 `palmprint_generator/third_party/opencv/build/`。已有安装包时加 `-Installer <exe> -SevenZip <7zr.exe>` 跳过下载。

然后编译：

```powershell
cmd /c palmprint_generator\build.bat
```

`build.bat` 用 `vswhere` 找最新的 VS，调 `vcvars64.bat`，编出 `deployment/palmprint.exe`，删掉 `.obj`。DLL 不在 `deployment/` 时，在 exe 旁边建一个指向 OpenCV 目录里那一份的软链接（建不了软链接时才复制），成功打印 `BUILD_OK`。OpenCV 放在别处时先设 `OPENCV_DIR` 指向官方包里的 `build` 目录。

`third_party/`、`deployment/` 下的 exe/dll/obj 和 `deployment/output/` 都在 `palmprint_generator/.gitignore` 里。OpenCV 不进 git，换机器跑一次 `fetch_opencv.ps1` 即可。

旧的 `%TEMP%\fp-deps\build_palm.bat` 不再需要，可以连同临时目录一起删掉。

运行，工作目录是 `deployment/`，因为 exe 旁边要有 `opencv_world454.dll`：

```powershell
.\palmprint.exe -o output\palm_left.jpg -seed 1 -layers output\layers_left
.\palmprint.exe -o output\palm_right.jpg -seed 1 -right
.\palmprint.exe -o output\palm_500.jpg -dpi 500 -seed 1
.\palmprint.exe -o output\palm_jp.jpg -seed 1 -population japanese
```

stdout 每次都打印 `population`、`shape`（含 `sex`、手长、掌长、掌宽、中指长、atd）、`lines`、`fingers`、`palm`（大鱼际、I2–I4、小鱼际）和 `pores`（总数与末节密度）几行，按特征挑种子靠它们。

`-layers` 写出 `regions.png`、`orientation.png`（HSV，色相是脊线方向）、`period.png`、`ridges.png`、`creases.png`、`palmprint.jpg`。旧的 `frequency.png` 已废弃。

## 管线

`PalmPrint::generate()` 按这个顺序走，全部在 `SourceCode/C++/palm.cpp`。

1. `build_geometry`：厘米坐标，y 向下，拇指在左。五指的根、轴向、长度、宽度写死，再乘一点种子抖动。手形见下面“手形”一节。末节纹型按人群预设的按指概率表抽（见“人群预设”）。指根三角 `tri_[0..3]` 在各指根下方 0.8 cm、指轴上；轴三角 `tri_[4]` 定在使 atd 约 40° 的位置。
2. `rasterize`：把区域画进 `labels_`（`region * 8 + finger`，手掌的 finger 是 7）。大鱼际、小鱼际的椭圆先画再和手掌多边形求交，不能画出轮廓。指根往掌心伸 0.7 cm 的一条，标成指根区（拇指标成大鱼际），用来接上手指和手掌。关节纹和主线画进 `creases_`，每点一个圆，半径沿曲线变化。主线见下面“掌部主线”一节。
3. `build_orientation`：`orientation_` 存的是脊线方向，不是法线。手指各节用 `finger_field`，掌部五区用 `palm_field`，再按模糊后的区域掩膜在双角（`2θ`）空间混合。最后加一幅低频扰动。
4. `build_period`：脊距像素。生理脊距 `kRidgePeriodCm = 0.048`（0.48 mm），换算后不低于 `kMinPeriodPx = 5`。脊距再乘性别系数（女 0.92），各区域再乘系数：末节 0.95、中节 1.0、基节 1.05、指根区 1.2、大鱼际和小鱼际 1.25、掌心 1.27、腕部 1.2（掌部脊线比指尖粗约 20–30%，文献是掌部脊密度约 10–12、指尖约 13–16），加 7% 低频噪声。上下限 0.85–1.5 倍基准，`kPeriods = 8`。
5. `render_ridges`：见下一节。
6. `composite`：脊线值映成灰度墨，乘油墨颗粒和边缘衰减。褶皱带上用低频噪声抠掉一圈，让脊线断头参差不齐。汗孔见下面“汗孔”一节。`-right` 时所有层水平翻转，方向场取负。

## 脊线渲染

这是这次改动的核心，也是最容易改坏的地方。

`render_ridges` 从 `[-1, 1]` 均匀噪声出发，做 10 轮。每轮对每个手掌像素做一次定向 Gabor 卷积，再 `tanh` 压回。褶皱和背景像素每轮清零，所以脊线不会穿过褶皱。

核的参数（`kAngles = 32`，角度只取 `[0, π)`，因为脊线无向；`kPeriods = 8`）：

- 方向：`ridge + π/2` 传给 `cv::getGaborKernel`。OpenCV 的核在 `xr` 轴上振荡，`xr` 垂直于纹路，所以要加 `π/2` 才是沿脊线。
- `sigma = 0.42 * period`，`gamma = 0.55`，半宽 `ceil(2.4 * sigma / gamma)`。`gamma < 1` 让核沿脊线拉长，线条才能连上。
- 核减去均值，避免每轮整体漂移。

方向场约定：

- 末节：`singularity()`，和 `fp::generate_orientation` 同一公式，`θ += atan2(core - p) / nCore`，`θ -= atan2(delta - p) / nDelta`，再乘 `nDelta / 2`。弧和帐用弓形公式。
- 末节修饰（`finger_field` 在 `singularity()` 之上叠加，公式本身没动）：
  - 箕、斗、双箕在核心以上 0.2 cm 起，用 smoothstep 叠一项 `tip_arch * sin(u·π/2)`（0.55–0.8 rad），让指尖脊线顺着轮廓成拱。符号和弓形分支一致。
  - 斗：`twist * exp(-r / 0.35)`，|twist| 0.2–0.45 rad，旋向随机。在圆环角上加常数偏移就是螺旋；不要接近 π/2，否则会变成放射线。
  - 双箕：两个核心横向各偏 0.17 倍指宽、沿轴错开 0.1 倍末节长。原来只偏 0.10 倍指宽时，看起来是同心椭圆。
  - 手指三节都叠了 3 个正弦细扰动（波长 0.6–1.2 cm，单个 0.02–0.035 rad）。
  - 这些参数用独立的 `cv::RNG tips`（`mix_seed(seed + 2^32)`），不改 `rng_` 序列，所以掌部不变。
  - stdout 的 `fingers ...` 一行打印各指纹型。
- 中节：垂直于指轴，加 `slope` 和一点侧向弯曲。
- 基节：同样的横纹，再加 `prox_bow * u`，所以是 V 或 Λ。
- 指根：Sherlock–Monro，4 个三角，每个配一个在其上方约 3.2 cm 的虚核心。指间区的箕（I2/I3/I4，按预设概率）在对应指缝中点下方 0.8 cm 放一个核心、1.4 cm 放一个三角。两个指根三角之间的脊线本来就竖着流进指缝，所以“核心在上、三角在下”这一对会形成朝指缝开口的箕；同一对放在横向背景里只会形成帐弓状，不要挪到横向流的区域。
- 大鱼际：默认绕拇指根的弧；按预设的 Th/I1 概率（`thenar_whorl_`）绕 `thenar_core_` 的拉长椭圆。正圆靶心看起来假，不要改回去。
- 小鱼际：绕掌外一点的弧；按预设的小鱼际概率出箕，箕型时圆心靠近尺侧。
- 掌心：单一斜角 `central_angle_`（约 0.30–0.52 rad）。
- 腕部：腕三角加一个在其下方的虚核心。

## 手形

坐标有三层：

1. **设计坐标**：代码里写死的手掌常量（外框 `kPalmOutline`、大小鱼际椭圆、指根区和腕区多边形、主线控制点、鱼际中心等）都写在这里，是最初那只手的坐标。
2. **模型坐标**：设计坐标经 `anat()` 得到，横向放宽 1.15 倍，指根以下（设计 y > 10.5）纵向拉长 1.31 倍。手指直接在模型坐标里建（指根位置经 `anat()`，宽度、长度直接写模型值），不经 `anat()`，所以手指不会被拉长。标准手（所有手形系数为 1）约为手长 18.2 cm、掌长 11 cm、掌宽 8.1 cm、中指 7.2 cm（`kStandardHand`、`kStandardBreadth`）。改之前的手掌长只有 8.8 cm、掌宽 7 cm，手指相对太长。
3. **图像坐标**：模型坐标经随种子的 `warp` 再乘 dpi。`to_px` / `to_cm` 负责这一层。

随种子的手形参数来自 `shape` 随机数流，stdout 的 `shape ...` 一行会打印出来：

- `sex`：男女各半。手长取预设的男/女均值加正态抖动（预设的标准差，截断在 ±2.5 SD），`size = 手长 / kStandardHand`。
- `width`：按预设的男/女“掌宽/手长”比例算出，再乘 0.95–1.05。`length` 0.96–1.04。
- `taper` ±0.025 /cm：只在掌部那一段起作用（模型 y 约 9 到 21.2，`kTaperFrom/To`），手指和腕部取两端的值，不会被挤扁。
- `spread` 0.75–1.3：四指绕中指轴张开或并拢。
- `thumb` −0.15 到 +0.02 rad：拇指绕根部转动，以外展为主。往掌心收超过约 +0.03 时，拇指尺侧边会压到生命线上段。
- 各指长度再乘 0.95–1.05（和原来 `rng_` 的 ±4% 叠加）。
- 打印的 `hand palm breadth middle atd` 是两层变换之后量出来的：手长 = 腕横纹到中指尖，掌长 = 中指根到腕横纹，掌宽 = 掌骨头高度的宽度，atd = 轴三角处 a、d 两三角的夹角。

150 个种子（generic）的实测：手长 18.2 ± 1.2 cm，掌长 11.0 ± 0.7，掌宽 8.1 ± 0.7，中指 7.2 ± 0.5，atd 38.7 ± 3°。文献：ANSUR II 中位数手长男 19.3 / 女 18.0 cm、掌宽男 8.8 / 女 7.8 cm；成人 atd 约 41°（SD 6–10）。

实现要点：

- 方向场在模型坐标里算，再用 `warp_angle` 按形变的雅可比矩阵转到图像里，所以形变后脊线仍然连续。
- 脊距是生理值，不随形变缩放：手大了，脊线条数就多。
- 椭圆用 `fill_ellipse`：圆心和半轴是设计坐标，逐点过 `anat()` 和 `warp`。不要用 `cv::ellipse` 直接画，它只会平移，不会跟着变形。
- 宽度类的量（指尖圆帽、褶皱半径）用 `px_len`，按 `size·sqrt(width·length)` 缩放。
- 主线起点的虎口点 `radial_web` 每次从拇指尺侧边和掌部桡侧边的交点算，不再写死；生命线、头线下笔前检查起点在手外、第二个点在掌上（不在手指上），否则在 stdout 打 `warning`。测试里 300 个种子不允许出现这个警告。

## 掌部主线

`add_principal_lines()` 按种子从几种解剖布局里抽，抽到的结果打印在 stdout 的 `lines ...` 一行，找特定变体的种子靠它。

- 生命线：从食指根和虎口之间的桡侧缘起笔，绕大鱼际，最宽处设计坐标 x≈4.65–5.25 cm。终点三选一：腕部（55%）、折回拇指根（25%）、短止（20%）。25% 带一条朝腕中部的分叉。
- 头线：按预设的 `separate`（现在各预设都是 30%，文献 26–35%，定义差异大）在生命线上方单独起笔，其余与生命线共起点、共走约 1 cm。单独起笔点在桡侧缘约 1/3 处，再往上会在手指张开时落到食指根上。末端：Sydney 线按预设概率延伸到尺侧缘，剩下的按 53:47 分给止于无名指下和下垂进小鱼际。
- 心线：从尺侧缘设计坐标 y≈11.0–11.5 起笔。终点：食指和中指之间（40%）、弯上食指根（25%）、直到桡侧（15%）、停在中指下（20%）。头线单独起笔时只取第一种和最后一种，避免和头线挤在食指下。35% 末端带分叉。
- 通贯掌按预设概率：一条横纹代替心线和头线。
- 命运线 40%（一半断成两段），太阳线 15%。腕横纹固定有。

实现要点：

- 曲线是 Catmull-Rom 过控制点（`add_curve`），每像素一个采样点，所以白带是连续的；断头效果全靠 `composite` 的 ragged 带。旧的三点 Bezier `add_tapered` 只剩指关节纹在用。
- 起点放在轮廓外约 0.25 cm，靠 `creases_.setTo(0, labels_ == 0)` 裁掉。
- 主线、腕横纹和细碎皱纹用独立的 `cv::RNG lines`，种子先经 splitmix 打散。不打散时 `cv::RNG` 对相邻种子的前几个输出高度相关，种子 8–25 几乎全抽到同一种生命线。独立 RNG 让初始噪声不随模板变；但褶皱每轮都会清零 `active`，所以改动的褶皱附近约 1 cm 内的脊线仍会变，全局 `gain` 也会被轻微牵动。
- 相邻主线保持 ≥ 0.4 cm，否则 ragged 外扩后会糊成一片白。
- 拇指根关节纹（`add_joint_crease(..., reach_across)`）尺侧只伸到 0.45 倍指宽，其余手指仍是 0.70。伸到 0.70 时它会接上生命线，看起来像一条横穿大鱼际的额外主线。

## 人群预设

`-population <name>`，默认 `han`。数据在 `SourceCode/C++/population.h`，一个预设一行结构体：按指纹型表（尺侧箕、桡侧箕、斗、双箕、弓、帐弓，拇指到小指）、掌部纹型（Th/I1、I2、I3、I4、小鱼际，按掌）、通贯掌、Sydney 线、头线单独起笔，以及男/女的手长均值和 SD、掌宽、脊距系数、汗孔格距。

| 预设 | 指纹 | 掌纹 | 手部尺寸 | 用了替代值的地方 |
| --- | --- | --- | --- | --- |
| `han` | 姜炳正等 2011，冀南汉族 n=1020（人类学学报 30(4)） | Zhang et al. 2010 PLoS One，全国汉族 n=11253；通贯掌按掌 3.23%（姜 2011） | GB 10000-1988 P50（转引），SD 由 P5–P95 推算 | Sydney 线用姜 2011 的“中贯型” 3.0%，未确认等同 |
| `european` | Buchwald 1981，北波兰 n=1234 | 美国白人（经 Zhang 2010 转引） | ANSUR II（美国陆军）中位数 | 通贯掌、Sydney 线用 generic；欧洲本土手部尺寸没查到（DINED 要登录，DIN 33402 付费） |
| `japanese` | 草加 1957，冈山备前 n=6556（表值 ×5 还原为每指频率） | Tomita 1994 范围中点 | AIST 日本人手部尺寸（L01、B03） | 通贯掌、Sydney 线用 generic |
| `generic` | 跨人群混合 | 跨人群混合 | ANSUR II | 不代表任何单一人群 |

所有预设共用的假设：
- 中国和日本的原始数据没有把双箕从斗里、帐弓从弓里拆出来，借用 Buchwald 1981 的拆分比例：双箕占斗，拇指 58%、示指 42%、中指 44%、环指 21%、小指 51%；帐弓占弓，依次是 3%、30%、23%、14%、13%。Buchwald 原文的 W² 是“二中心斗”，和双箕是否完全等同没有核实。
- 三个人群都没查到汗孔密度，脊距和汗孔格距按性别用同一组值（女：脊距 ×0.92，格距 0.0262 cm；男：×1.0，0.0303 cm）。格距是标定出来的，使 500 dpi 末节密度落在女约 52、男约 45 个/9 mm²（印度样本的文献值）；脊距系数依据指尖脊密度女高于男约 10%。
- 头线单独起笔的比例各预设都用 30%。

完整出处链接见本节末尾。所有引用来自网页检索的开放获取页面；组织要求的 LibKey 学术连接器在会话里没有登录，没有经过 LibKey 解析。

- 姜炳正等 2011：http://www.ivpp.cas.cn/cbw/rlxxb/xbwzxz/201112/P020111222353685966378.pdf
- Zhang et al. 2010：https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0008783
- GB 10000-1988 转引：https://pmc.ncbi.nlm.nih.gov/articles/PMC9225906/
- Buchwald 1981：https://czasopisma.uni.lodz.pl/ar/article/view/21314
- 草加 1957：https://ousar.lib.okayama-u.ac.jp/42600
- Tomita 1994：https://pubmed.ncbi.nlm.nih.gov/7807711
- AIST 手部尺寸：https://www.airc.aist.go.jp/dhrt/hand/data/list.html
- ANSUR II 汇总表：https://ergocenter.ncsu.edu/wp-content/uploads/sites/358/2026/02/01-Anthropometry-Summary-Tables.pdf
- 汗孔面密度（Kaur & Dhall 2024）：https://www.crime-scene-investigator.net/PDF/epidermal-ridge-sweat-pore-density-a-forensic-approach-to-sex-determination.pdf
- 掌部与指尖脊密度：https://www.jkimsu.com/jkimsu-vol9no1/JKIMSU,%20Vol.%209,%20No.%201,%20January-March%202020%20Page%2051-57.pdf ；https://doi.org/10.2478/anre-2020-0025
- atd、a-b 嵴数、掌纹（比利时 1979）：https://biblio.naturalsciences.be/associated_publications/anthropologica-prehistorica/bulletin-de-la-societe-royale-belge-d-anthropologie-et-de-prehistoire/ap-090-1979/ap90_99-114.pdf

## 汗孔

在 `composite` 末尾放。把画布切成边长为预设汗孔格距（约 0.26–0.30 mm）的格子，格子边长是小数像素（边界取 `floor(i·cell)`；取整成整数像素时，500 dpi 下 0.028–0.031 cm 全变成 6 px，男女格距就没有区别了），每格保留概率 80%；在格内到脊线边缘距离 ≥ 0.7 倍格内最大值的像素里随机挑一个作为汗孔，再检查和左、左上、上、右上四格已放的汗孔距离 ≥ 0.6 格。500 dpi 时汗孔半径 1 px，四分之一概率 2 px；250 dpi 只把那个像素压暗。

为什么这样放：按网格取每格脊线正中心，汗孔会沿脊线排成一串整齐的虚线，看起来像缝线；改成在靠近中心的像素里随机挑以后才不规整。

stdout 的 `pores count=... distal=.../9mm2` 是程序自己数的总数和末节（扣掉褶皱）面密度。500 dpi 下末节女约 53、男约 46 个/9 mm²，文献成人男 45 / 女 52。防粘连规则让密度随格距缩小增长得比 1/格距² 慢，重新标定时要实测。250 dpi（默认）下只有约 18–24，因为脊线被 5 px 下限撑宽、可放的中心像素少，这是分辨率的限制；脊距低于 8 px（约 420 dpi 以下）时程序会在 stdout 打一行 `note:` 提醒，`-h` 里也写了。做汗孔相关的数据一定要加 `-dpi 500`。默认 dpi 仍是 250（500 dpi 单张约 12–15 s，250 dpi 约 1.5 s）。

## 不要再走的路

下面几种渲染都做过，输出是噪声或虚线。方向场本身不是问题。

- 二值化 Gabor：碎成噪点。
- 沿用指纹生成器那套短 `sigma` 的连续 Gabor：核太短，脊线接不上。
- 泊松相位积分：积不出稳定条纹。
- 沿方向场追踪流线：断、乱。
- 按区域盖章（横线、同心椭圆）：不读方向场，掌心过于规则，放大是虚线。

`labels_` 上不要用 `LINE_AA`。抗锯齿会写出 0–255 之间的垃圾标签，`finger_of` 会拿到 7 以外的值，随后越界访问 `fingers_[5]`，退出码 `-1073741819`。标签图层用 `LINE_8`，并且在读手指下标时确认 `< 5`。

## 已知限制

- 5 px 是脊距下限。低于约 265 dpi 时脊线比真人粗、条数少。要真实比例用 `-dpi 500`。帮助文本里写了。
- 末节仍是原仓库的箕斗模型加修饰，没有真实指纹里的疤痕、褶皱和局部纹型变异。
- 主线只有上面那几种模板，指关节纹仍是固定位置加抖动。
- 右手是同一种子左手的逐像素镜像，不是“同一个人的另一只手”。要做成后者，需要给右手单独抽细节，同时保留手形和纹型分布上的相似性；还没做。
- 生命线上段一直紧贴拇指尺侧边（固定手形时就是这样），拇指往掌心收的范围因此只放到 +0.02 rad。
- 汗孔在 `base_period_ < 8`（即默认 250 dpi）时只打 1 像素、密度也只有 500 dpi 的一半左右；要汗孔真实用 500 dpi。
- 人群数据有缺口（见“人群预设”）：欧洲手部尺寸用美国数据代替，日本和欧洲的通贯掌、Sydney 线用 generic，三个人群都没有汗孔数据，双箕和帐弓的拆分借用波兰数据。
- 没有模拟平面捺印时掌心凹陷处接触不良、留白的情况；现在是全掌均匀着墨，更像滚印。
- 指间区的箕开口朝尺侧偏指缝，形态说得过去，但位置和大小是固定的（`kLoopCoreDepth`、`kLoopDeltaDepth`），没有按种子变化。

## 建议的下一步

按影响从大到小：

1. 右手单独生成（见已知限制），取决于数据集要“一人两手”还是“左右镜像”。
2. 补人群数据缺口，尤其是欧洲手部尺寸和三个人群的汗孔密度。
3. 可选的“平面捺印”模式：掌心凹陷处按压力衰减留白。
4. 末节加疤痕、白线（细褶皱）这类个体特征。

## 不要提交

`deployment/` 里的 `palmprint.exe`、`opencv_world454.dll`、`*.obj`，`deployment/output/` 的图片，以及 `third_party/`，都是本地产物，已经写进 `.gitignore`。原仓库 `gpu_fingerprint_generator/deployment/` 下还有一批为了跑旧 demo 放进去的 DLL（MinGW 运行库、`cudart64_110.dll`、OpenCV debug 转发 DLL），同样不是源码。
