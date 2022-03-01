#!/usr/bin/env python
# coding=utf-8

'''
繁体与简体相互转换测试：
 '''

import changeCode

if __name__ == '__main__':
    # simple_str = '临 为丽举么义乌乐乔习乡书买乱争于亏云亘亚nhao你好点发货ID的疯狂大幅度发'
    # print('   简体原文：%s' % simple_str)
    # to_tradition_str = changeCode.toTraditionString(simple_str)
    # print('简体转换繁体：%s' % to_tradition_str)
    #
    # tradition_str = 'ni好地價眾優傴傘方y地方和夥會'
    # print('   繁体原文：%s' % tradition_str)
    # to_simple_str = changeCode.toSimpleString(tradition_str)
    # print('繁体转换简体：%s' % to_simple_str)

    text_path = "input_zh_size1.txt"
    output_path = "input_zh_simplified.txt"
    with open(text_path, "r", encoding="utf-8") as file:
        data = file.read()
        simplified_text = changeCode.toSimpleString(data)
        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.write(simplified_text)
            outfile.flush()
