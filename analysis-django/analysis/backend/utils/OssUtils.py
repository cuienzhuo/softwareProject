import oss2
from aliyunsdkcore.acs_exception.exceptions import ClientException, ServerException
import os

class OssUtils:

    def upload_buffer_to_oss(self,buffer, oss_config, oss_file_path):
        try:
            # 1. 初始化OSS认证和Bucket客户端
            auth = oss2.Auth(
                oss_config["access_key_id"],
                oss_config["access_key_secret"]
            )
            bucket = oss2.Bucket(
                auth,
                oss_config["endpoint"],
                oss_config["bucket_name"]
            )

            # 2. 从buffer上传图片（指定MIME类型为image/png）
            bucket.put_object(
                key=oss_file_path,  # OSS中的文件路径
                data=buffer,  # 内存缓冲区数据
                headers={"Content-Type": "image/png"}
            )

            # 3. 生成公开访问URL（若Bucket为私有，需生成临时URL）
            # 公开Bucket的URL格式：https://BucketName.Endpoint/ObjectName
            public_url = f"https://{oss_config['bucket_name']}.{oss_config['endpoint']}/{oss_file_path}"
            return public_url

        except ClientException as e:
            return f"OSS客户端错误: {str(e)}（可能是AccessKey或Endpoint错误）"
        except ServerException as e:
            return f"OSS服务端错误: {str(e)}（可能是Bucket不存在或权限不足）"
        except Exception as e:
            return f"上传失败: {str(e)}"

    def OssUpload(self,buffer,save_path):
        # 1. 配置你的OSS信息（必须替换为实际值）
        OSS_CONFIG = {
            "access_key_id": os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID"),  # 读取环境变量
            "access_key_secret": os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),  # 读取环境变量
            "endpoint": "oss-cn-beijing.aliyuncs.com",
            "bucket_name": "qiluyiyuan"
        }
        if not OSS_CONFIG["access_key_id"] or not OSS_CONFIG["access_key_secret"]:
            raise ValueError("请先设置环境变量 ALIBABA_CLOUD_ACCESS_KEY_ID 和 ALIBABA_CLOUD_ACCESS_KEY_SECRET")

        # 3. 执行上传并获取URL（假设buffer已准备好）
        image_url = self.upload_buffer_to_oss(buffer, OSS_CONFIG, save_path)
        return image_url