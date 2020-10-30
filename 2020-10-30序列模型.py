'''
集束搜索（Beam Search):
    贪婪算法只会挑出最可能的那一个单词，然后继续。而集束搜索则会考虑多个选择，集束搜索算法会有一个参数B，叫做集束宽（beam width）
    注意如果集束宽等于1，只考虑1种可能结果，这实际上就变成了贪婪搜索算法，上个视频里我们已经讨论过了。但是如果同时考虑多个，可能的
    结果比如3个，10个或者其他的个数，集束搜索通常会找到比贪婪搜索更好的输出结果

改进集束搜索（Refinements to Beam Search）:
    归一化的集束搜索

集束搜索的误差分析（Error analysis in beam search）:

Bleu 得分:

注意力模型直观理解（Attention Model Intuition）:

注意力模型（Attention Model）:
    注意力模型如何让一个神经网络只注意到一部分的输入句子。当它在生成句子的时候，更像人类翻译

语音识别（Speech recognition）:

触发字检测（Trigger Word Detection）:

'''