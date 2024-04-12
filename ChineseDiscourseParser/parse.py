#!/usr/bin/env python3
# coding: UTF-8
import argparse
from structure import Discourse
from pipeline import build_pipeline
import logging
import tqdm
import os,sys





# pipeline = build_pipeline(schema="topdown", segmenter_name="svm", use_gpu=False)
def run(args):
    logger = logging.getLogger("dp")
    doc = Discourse()
    pipeline = build_pipeline(schema=args.schema, segmenter_name=args.segmenter_name, use_gpu=args.use_gpu)
    with open(args.source, "r", encoding=args.encoding) as source_fd:
        for line in tqdm.tqdm(source_fd, desc="parsing %s" % args.source, unit=" para"):
            line = line.strip()
            if line:
                para = pipeline(line)
                # if args.draw:
                #     para.draw()
                #这里得到的应该已经是tree了，进行遍历即可
                doc.append(para) #构造完成得到tree，进行处理
    #此处doc包含了所有的段落，直接建树
        # print(doc,type(doc[0]))
    # logger.info("save parsing to %s" % args.save)
    return doc
    # doc.to_xml(args.save, encoding=args.encoding)


# def test():
#     import pyltp
#     pyltp.Postagger()
#     print('ll')
def main():
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source",default="./sample.txt")
    arg_parser.add_argument("--save",default='./sample1.xml')
    arg_parser.add_argument("--schema", default="topdown")
    arg_parser.add_argument("--segmenter_name", default="svm")
    arg_parser.add_argument("--encoding", default="utf-8")
    arg_parser.add_argument("--draw",default=False, dest="draw", action="store_true")
    arg_parser.add_argument("--use_gpu",default=False, dest="use_gpu", action="store_true")
    arg_parser.set_defaults(use_gpu=False)
    arg_parser.set_defaults(draw=False)
    args=arg_parser.parse_args()
    run(args)
if __name__ == '__main__':

    # test()
    main()


