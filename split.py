import pydub
import numpy as np
import pandas as pd
from tqdm import tqdm
from pydub.utils import which


def split(filepath):
    sound = pydub.AudioSegment.from_wav(filepath)
    dBFS = sound.dBFS
    tmp_chunks = pydub.silence.split_on_silence(sound,
                                                min_silence_len=500,
                                                silence_thresh=dBFS - 16)
    chunks = []
    label = 0
    count = 0
    for i in range(len(tmp_chunks)):
        if label != 0:
            label = 0
            continue
        if len(tmp_chunks[i]) < 4000 and i < len(tmp_chunks) - 1:
            chunks.append(tmp_chunks[i] + tmp_chunks[i + 1])
            label = 1
            count += 1
        else:
            chunks.append(tmp_chunks[i])
    return chunks


# version2
def split2(filepath):
    sound = pydub.AudioSegment.from_file(filepath, format="mp3")
    dBFS = sound.dBFS
    chunks, times = pydub.silence.split_on_silence(sound,
                                                   min_silence_len=500,
                                                   silence_thresh=dBFS - 17)  # 17是气声不算静音
    for j in range(5):
        for i, chunk in enumerate(chunks):
            # 如果小于4s要合并
            if len(chunks[i]) < 4000:
                if i != len(chunks) - 1:
                    if len(chunks) >= 2:
                        # 如果不是最后一个就往右并
                        chunks.insert(i, chunks.pop(i) + chunks.pop(i))
                        f_time = times.pop(i)
                        b_time = times.pop(i)
                        times.insert(i, (f_time[0], b_time[1]))
                else:
                    if len(chunks) >= 2:
                        # 如果是最后一个就往左并
                        back = chunks.pop()
                        front = chunks.pop()
                        chunks.append(front + back)

                        b_time = times.pop()
                        f_time = times.pop()
                        times.append((f_time[0], b_time[1]))

            # 如果大于20s要分的更细并插入回数组中
            elif len(chunks[i]) > 20000:
                tmp = chunks.pop(i)
                tiny_chunks, tiny_times = pydub.silence.split_on_silence(tmp, min_silence_len=300,
                                                                         silence_thresh=tmp.dBFS - 14)
                tiny_chunks.reverse()
                for tiny_chunk in tiny_chunks:
                    chunks.insert(i, tiny_chunk)

                tiny_times.reverse()
                for tiny_time in tiny_times:
                    times.insert(i, tiny_time)
    return chunks, times

pydub.AudioSegment.converter = which("ffmpeg")
wav_list = open("./vocal_wav_list.txt", "r")
lines = wav_list.readlines()
df_index = 0
df = pd.DataFrame({'name': "test", 'num': 0, 'time_start': [0], 'time_end': [0]}, index=[df_index])
for line in tqdm(lines):
    tmp = line.rstrip()
    chunks, times = split2(tmp)
    for i in range(len(chunks)):
        des = tmp.rstrip("vocals.wav")
        des = des.rstrip('/')
        name = des[des.rfind('/') + 1:]

        df_index += 1
        df = df.append(
            pd.DataFrame({'name': name, 'num': i, 'time_start': times[i][0] / 1000, 'time_end': times[i][1] / 1000},
                         index=[df_index]))

        chunks[i].export("{}/split/{}.wav".format(des, i), "wav")
df.to_csv('./times_list.csv', index=False)
