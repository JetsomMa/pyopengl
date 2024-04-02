"""
这个功能的作用： 
    设置好一个要加载的obj模型，
    相机会自动切换不同的位置【实际发现以更改相机位置的方式，在相机掠过南北极位置时，拍照结果不符合预期，所以采用了旋转模型的方式】
    自动生成每个角度下的纹理图、深度图、法线数值图、线框图
"""
import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import gluLookAt, gluPerspective
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

class Camera:
    # pitch（俯仰角） -90 ～ 90
    # yaw（偏航角）0 - 360
    def __init__(self, reference_point, pitch, yaw, distance):
        self.reference_point = reference_point
        self.pitch = pitch
        self.yaw = yaw
        self.distance = distance

# 全局变量存储顶点和面
vertices = []
faces = []
texture_coords_indexs = []
texture_coords = []
normals = []
texture_id = None  # 纹理ID
center = (0, 0, 0)  # 模型中心点
max_size = 10  # 模型最大尺寸
width, height = 1500, 1500
camera = Camera([0, 0, 0], 0, 0, 10)

def set_camera_lookat(reference_point, pitch, yaw, distance):
    # 将角度从度转换为弧度
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    
    # 计算相机位置
    camera_x = reference_point[0] + distance * np.cos(pitch_rad) * np.sin(yaw_rad)
    camera_y = reference_point[1] + distance * np.sin(pitch_rad)
    camera_z = reference_point[2] + distance * np.cos(pitch_rad) * np.cos(yaw_rad)
    
    # 上向量，这里我们假设相机不会绕x轴旋转（即没有横滚），因此上向量始终为(0, 1, 0)
    up = (0, 1, 0)
    
    # gluLookAt的入参
    eye = (camera_x, camera_y, camera_z)
    center = reference_point
    
    return eye, center, up

def set_camera_lookat2(reference_point, pitch, roll, distance):
    # 将角度从度转换为弧度
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    
    # 计算相机位置
    camera_x = reference_point[0] + distance * np.sin(pitch_rad)
    camera_y = reference_point[1] + distance * np.cos(pitch_rad) * np.sin(roll_rad)
    camera_z = reference_point[2] + distance * np.cos(pitch_rad) * np.cos(roll_rad)
    
    # 计算上向量，假设在横滚角为0时，上向量为(0, 1, 0)
    up_x = -np.sin(roll_rad)
    up_y = np.cos(roll_rad)
    up_z = 0  # 在此简化情景下，我们假定相机的横滚不会影响Z轴分量
    
    # gluLookAt的入参
    eye = (camera_x, camera_y, camera_z)
    center = reference_point
    up = (up_x, up_y, up_z)
    
    return eye, center, up

def load_obj(filename):
    vertices = []
    faces = []
    texture_coords_indexs = []
    texture_coords = []
    normals = []
    mtl_path = None

    def calculate_normal(v1, v2, v3):
        # Calculate vectors
        u = [v2[i] - v1[i] for i in range(3)]
        v = [v3[i] - v1[i] for i in range(3)]
        # Calculate cross product
        normal = [(u[1]*v[2] - u[2]*v[1]), (u[2]*v[0] - u[0]*v[2]), (u[0]*v[1] - u[1]*v[0])]
        # Normalize the normal
        length = (normal[0]**2 + normal[1]**2 + normal[2]**2)**0.5
        return [normal[0]/length, normal[1]/length, normal[2]/length]

    # 初始化用于计算边界的变量
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append((x, y, z))

                # 更新边界
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_z, max_z = min(min_z, z), max(max_z, z)
            elif line.startswith('f '):
                parts = line.split()
                face = []
                face_tex_coords = []
                for part in parts[1:]:
                    values = part.split('/')
                    face.append(int(values[0]) - 1)
                    if len(values) > 1 and values[1]:
                        face_tex_coords.append(int(values[1]) - 1)
                faces.append(face)
                texture_coords_indexs.append(face_tex_coords)
                # Calculate normal for this face
                normal = calculate_normal(vertices[face[0]], vertices[face[1]], vertices[face[2]])
                normalized = [(component + 1) / 2 for component in normal]
                normals.append(normalized)
            elif line.startswith('vt '):
                parts = line.split()
                texture_coords.append((float(parts[1]), float(parts[2])))
            elif line.startswith('mtllib '):
                parts = line.split()
                mtl_path = parts[1]

    # 计算中心点
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    center = (center_x, center_y, center_z)

    # 计算最大尺寸
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    max_size = max(size_x, size_y, size_z)

    return vertices, faces, texture_coords_indexs, texture_coords, normals, mtl_path, center, max_size

def load_mtl(full_path):
    texture_path = None
    with open(full_path, 'r') as file:
        for line in file:
            if line.startswith('map_Kd'):
                parts = line.split()
                texture_path = texture_path = parts[1]
    return texture_path

def load_texture(texture_path):
    img = Image.open(texture_path)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.array(list(img.getdata()), np.uint8)

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.size[0], img.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    return texture_id

def create_black_texture():
    black_texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, black_texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    black_data = np.zeros((1, 1, 3), dtype=np.uint8)  # 创建1x1的黑色纹理数据
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, black_data)
    return black_texture_id

def init_gl(width, height):
    glutInit()
    # glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE)
    glutInitWindowSize(width, height)
    glutCreateWindow(b"OBJ Model Wireframe")
    glEnable(GL_DEPTH_TEST)
    
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (width/height), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

def render_scene(texture_id, render_type="wireframe", filename_split="", rotate = [0,0,0]):
    global vertices, faces, texture_coords_indexs, texture_coords, normals

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    eye, center, up = set_camera_lookat(camera.reference_point, camera.pitch, camera.yaw, camera.distance)
    gluLookAt(*eye, *center, *up)

    glTranslatef(camera.reference_point[0], camera.reference_point[1], camera.reference_point[2])
    # 参数分别为：绕X轴旋转角度，绕Y轴旋转角度，绕Z轴旋转角度
    glRotatef(rotate[0], 1, 0, 0) # 绕X轴旋转20度
    glRotatef(rotate[1], 0, 1, 0) # 绕Y轴旋转30度
    glRotatef(rotate[2], 0, 0, 1) # 绕Y轴旋转30度
    glTranslatef(-camera.reference_point[0], -camera.reference_point[1], -camera.reference_point[2])
    
    if render_type == "wireframe":
        texture_id = create_black_texture()

        # 使用纯黑色纹理填充
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glBegin(GL_TRIANGLES)
        for face in faces:
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()
        glDisable(GL_TEXTURE_2D)

        # 渲染线框
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glLineWidth(1.5)  # 设置较宽的线宽
        glColor3f(1.0, 1.0, 1.0)  # 黑色背景线框

        glBegin(GL_TRIANGLES)
        for face in faces:
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)  # 恢复为填充模式
        save_image(width, height, "image_" + filename_split + '_' + render_type + ".png")
    elif render_type == "depth":
        # 禁用颜色
        # glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)

        # 渲染几何体
        glBegin(GL_TRIANGLES)
        for face in faces:
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()

        # 恢复颜色写入
        # glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)

        # 读取并保存深度信息
        save_depth_image(width, height, "image_" + filename_split + '_' + render_type + ".png")
    elif render_type == "normal":
        glBegin(GL_TRIANGLES)
        for i, face in enumerate(faces):
            glColor3f(*normals[i])  # 黑色背景线框
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()
        save_image(width, height, "image_" + filename_split + '_' + render_type + ".png")
    else:
        if texture_id:
            glEnable(GL_MULTISAMPLE)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            glBegin(GL_TRIANGLES)
            glColor3f(1.0, 1.0, 1.0)  # 恢复为白色
            for i, face in enumerate(faces):
                for j, vertex in enumerate(face):
                    if texture_coords_indexs[i][j]:
                        glTexCoord2fv(texture_coords[texture_coords_indexs[i][j]])
                    glVertex3fv(vertices[vertex])
            glEnd()
            glDisable(GL_MULTISAMPLE)
            glDisable(GL_TEXTURE_2D)
        else:
            glBegin(GL_TRIANGLES)
            glColor3f(1.0, 1.0, 1.0)  # 恢复为白色
            for face in faces:
                for vertex in face:
                    glVertex3fv(vertices[vertex])
            glEnd()
        
        save_image(width, height, "image_" + filename_split + '_' + render_type + ".png")

def save_image(width, height, filename="output.png"):
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save('images/' + filename)

def save_depth_image(width, height, filename="depth.png"):
    depth_buffer = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)
    depth_array = np.frombuffer(depth_buffer, dtype=np.float32).reshape(height, width)
    depth_image = np.flipud(depth_array)  # 上下翻转图像

    # 反转深度值，使得近处为白色，远处为黑色
    depth_image = 1.0 - depth_image  # 反转深度值

    depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())  # 归一化
    depth_image = (depth_image * 255).astype(np.uint8)  # 转换为灰度值
    image = Image.fromarray(depth_image, 'L')
    image.save('images/' + filename)

# 选择模型
def select_model(root_dir, model_name):
    global vertices, faces, texture_coords_indexs, texture_coords, normals, texture_id, center, max_size
    model_root_dir = root_dir
    
    init_gl(width, height)

    obj_path = os.path.join(model_root_dir, model_name)
    vertices, faces, texture_coords_indexs, texture_coords, normals, mtl_path, center, max_size  = load_obj(obj_path)

    if mtl_path:
        texture_path = load_mtl(os.path.join(model_root_dir, mtl_path))
        texture_id = load_texture(os.path.join(model_root_dir, texture_path))
    else:
        texture_id = None

# 截取照片
def cropThePhoto(filename_split, rotate):
    render_type = "texture"
    render_scene(texture_id, render_type, filename_split, rotate)
    
    render_type = "wireframe"
    render_scene(texture_id, render_type, filename_split, rotate)

    render_type = "depth"
    render_scene(texture_id, render_type, filename_split, rotate)

    render_type = "normal"
    render_scene(texture_id, render_type, filename_split, rotate)

def initPath(folder_path):
    # 判断文件夹路径是否存在
    if not os.path.exists(folder_path):
        # 如果文件夹路径不存在，则创建文件夹
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def main():
    global camera
    # 选择模型
    select_model("./model/grondtruth_obj", "scene.obj")
    # select_model("./model/cow", "cow.obj")
    
    # 截取照片
    # pitch（俯仰角） -90 ～ 90
    # yaw（偏航角）0 - 360
    camera.reference_point = [center[0], center[1], center[2]]
    camera.pitch = 0
    camera.yaw = 0
    camera.distance = 1.8 * max_size
 
    angle = 5
    timer = int(360 / angle)

    print("center: ", center)
    print("max_size: ", max_size)
    print("timer: ", timer)

    initPath('images')

    rotate = [0, 0, 0]
    for i in range(1, timer + 1):
        rotate[0] += angle
        cropThePhoto('1-' + str(i), rotate)

    rotate = [0, 0, 0]
    for i in range(1, timer + 1):
        rotate[1] += angle
        cropThePhoto('2-' + str(i), rotate)

    rotate = [0, 0, 0]
    camera.yaw = camera.yaw + 90
    for i in range(1, timer + 1):
        rotate[2] += angle
        cropThePhoto('3-' + str(i), rotate)
        
    
if __name__ == "__main__":
    main()
