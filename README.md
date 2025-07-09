\# Coladan

A High-Performance, Reliable Multimodal-Multiomic Whole Slide AI Model generates Genome-wide Spatial Gene Expression from Histopathology Images



\# environment

```

git clone https://github.com/99wzj/Coladan.git

cd Coladan

conda create -n coladan python=3.10.0 -y

pip install --upgrade pip

conda install r-base

pip install -e .

\#install faster by no-build-isolation

pip install flash-attn==1.0.9 --no-build-isolation

```



\# for GPU version

pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/cu118/torch\_stable.html



\# if import error with"libstdc++" (by flash\_attn <2)

try  libstdc++.so.6.0.29

maybe in your conda environment

```

strings /usr/lib/x86\_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX

find / -name "libstdc++.so.6\*"

strings /root/miniconda3/envs/coladan/lib/libstdc++.so.6.0.29 | grep GLIBCXX

cp /root/miniconda3/envs/coladan/lib/libstdc++.so.6.0.29 /usr/lib/x86\_64-linux-gnu/

rm /usr/lib/x86\_64-linux-gnu/libstdc++.so.6

ln -s /usr/lib/x86\_64-linux-gnu/libstdc++.so.6.0.29 /usr/lib/x86\_64-linux-gnu/libstdc++.so.6

```



\# weight download and quick start demo

https://huggingface.co/WangZj99/Coladan





