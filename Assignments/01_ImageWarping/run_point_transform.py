import cv2
import numpy as np
import gradio as gr
import jax
import jax.numpy as jnp

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换
alpha = 1e3
eps = 1e-2

@jax.jit
def rbf_basis(xi, xj, d):
    return jnp.exp(-jnp.linalg.norm(xi - xj) ** 2 / d)

@jax.jit
def rbf_deform(query, points_src, points_dst, x):
    """
    query: (2, ) array
    points_src: (M, 2) array
    points_dst: (M, 2) array
    x: (M+3, 2) array, coefficient for each basis.

    Result.x = \sum_j x_j.x Basis_j(query)
    """

    M = points_src.shape[0]
    basis = jnp.array([rbf_basis(query, points_dst[j], alpha) for j in range(M)] + [query[0], query[1], 1])
    print(basis.shape, x.shape)
    out_x, out_y = jnp.sum(x[0, :] * basis), jnp.sum(x[1, :] * basis)

    return jnp.array([out_x, out_y])


def fit_rbf(point_src, point_dst):
    # fit the matrix
    M = point_dst.shape[0]
    A = np.zeros((M+3, M+3))
    A[:M, :M] = np.array([[rbf_basis(point_dst[i], point_dst[j], alpha) for j in range(M)] for i in range(M)])
    A[:M, M:M+2] = point_dst
    A[:M, M+2] = 1
    A[M:M+2, :M] = point_dst.T
    A[M+2, :M] = 1
    b = np.zeros((M+3, 2))
    b[:M] = point_src
    x = np.linalg.solve(A, b).T
    return A, x


def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    warped_image = np.array(image)
    w, h, _ = image.shape
    ### FILL: 基于MLS or RBF 实现 image warping
    # source_pts[:, 0], source_pts[:, 1] = source_pts[:, 1], source_pts[:, 0]
    # target_pts[:, 0], target_pts[:, 1] = target_pts[:, 1], target_pts[:, 0]
    source_pts = np.array(source_pts)
    source_pts = np.flip(source_pts, axis=1)
    target_pts = np.array(target_pts)
    target_pts = np.flip(target_pts, axis=1)


    A, x = fit_rbf(source_pts, target_pts)
    print(A, x)

    source_pts = jnp.array(source_pts.astype(np.float32))
    target_pts = jnp.array(target_pts.astype(np.float32))
    queries = jnp.array([[i, j] for i in range(w) for j in range(h)])
    out = jax.vmap(rbf_deform, in_axes=(0, None, None, None))(queries, source_pts, target_pts, x)
    out = np.array(out.reshape(w, h, 2).astype(np.int32))
    # clamp
    out = np.clip(out, 0, np.array([w-1, h-1]))
    warped_image = warped_image[out[:, :, 0], out[:, :, 1]]

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
