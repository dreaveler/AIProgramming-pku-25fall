import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = '1.0.0'

with open("README.md", "r", encoding="utf-8") as fid:
  long_description = fid.read()


dir = './csrc'
sources = ['{}/{}'.format(dir, src) for src in os.listdir(dir)
           if src.endswith('.cpp') or src.endswith('.cu')]

cuda_home = os.environ.get("CUDA_HOME", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0")
cuda_lib_dir = os.path.join(cuda_home, "lib", "x64")

setup(
    name='torchofdreaveler',
    version=__version__,
    author='Dreaveler',
    author_email='2169448673@qq.com',
    url='https://github.com/dreaveler/AIProgramming-pku-25fall.git',
    description='Homework of pku2025fall AIProgramming',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['torchofdreaveler'],
    # packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=['torch', 'numpy'],
    python_requires='>=3.8',
    license='MIT',
    ext_modules=[
        CUDAExtension(
            name='torchofdreaveler.core',
            sources=sources,
            libraries=['cudart','cublas','curand'],
            library_dirs=[cuda_lib_dir],
            include_dirs=[os.path.join(cuda_home, "include")]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    classifiers=[
        'License :: OSI Approved :: MIT classifiers',
    ],
)