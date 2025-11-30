import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import os
import shutil

# 初始化管道
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "/mnt/sda/ybc/models--stabilityai--stable-diffusion-xl-refiner-1.0/snapshots/5d4cfe854c9a9a87939ff3653551c2b3c99a4356", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True, 
    local_files_only=True
)
pipe = pipe.to("cuda:3")

# 配置路径
input_images_folder = "data_yolo_3073/images/train"  # 输入文件夹路径
input_labels_folder = "data_yolo_3073/labels/train"  # YOLO标签文件夹路径
output_images_folder = "data_train/images/train"  # 输出文件夹路径
output_labels_folder = "data_train/labels/train"  # 输出标签文件夹路径
prompt = "a satellite photo of an airplane, make sure all the aircrafts and characteristic details of the aircraft preserved compared to the original image"

# 创建输出文件夹
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

# 支持的图片格式
supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def process_images():
    # 获取输入文件夹中的所有图片文件
    image_files = [f for f in os.listdir(input_images_folder) 
                  if os.path.isfile(os.path.join(input_images_folder, f)) and 
                  os.path.splitext(f)[1].lower() in supported_formats]
    
    print(f"找到 {len(image_files)} 张图片需要处理")
    
    for i, image_file in enumerate(image_files):
        print(f"正在处理第 {i+1}/{len(image_files)} 张图片: {image_file}")
        
        # 构建完整路径
        image_path = os.path.join(input_images_folder, image_file)
        
        # 加载图片
        try:
            init_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"无法加载图片 {image_file}: {e}")
            continue
        
        # 生成新图片
        try:
            generated_images = pipe(prompt, image=init_image)
            generated_image = generated_images.images[0]
        except Exception as e:
            print(f"生成图片失败 {image_file}: {e}")
            continue
        
        # 处理文件名
        file_name, file_ext = os.path.splitext(image_file)
        new_image_name = f"{file_name}_generated{file_ext}"
        new_image_path = os.path.join(output_images_folder, new_image_name)
        
        # 保存生成的图片
        generated_image.save(new_image_path)
        print(f"已保存生成图片: {new_image_name}")
        
        # 复制原图片到输出文件夹
        original_copy_path = os.path.join(output_images_folder, image_file)
        shutil.copy2(image_path, original_copy_path)
        
        # 处理YOLO标签文件
        label_file = f"{file_name}.txt"  # YOLO标签文件通常与图片同名，扩展名为.txt
        label_path = os.path.join(input_labels_folder, label_file)
        new_label_name = f"{file_name}_generated.txt"
        new_label_path = os.path.join(output_labels_folder, new_label_name)
        
        # 如果存在标签文件，则复制并重命名
        if os.path.exists(label_path):
            shutil.copy2(label_path, new_label_path)
            print(f"已复制标签文件: {new_label_name}")
        
        # 复制原标签文件到输出文件夹（如果需要）
        original_label_copy_path = os.path.join(output_labels_folder, label_file)
        if os.path.exists(label_path):
            shutil.copy2(label_path, original_label_copy_path)
    
    print("所有图片处理完成！")

if __name__ == "__main__":
    process_images()