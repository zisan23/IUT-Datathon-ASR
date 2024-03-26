import random
import torch
import json
from .base_dataset import BaseDataset


class A_OKVQA_VQGA_EvalDataset(BaseDataset):
    VQGA_INSTRUCTIONS = (
        "Based on the content of the given image, generate a question that requires reasoning using a variety of knowledge types such as commonsense and then briefly answer it.",
    )
    VQGA_PROMPTS = (
        " Question: {q}",
    )

    def __init__(self, vis_processor, text_processor, vis_root, anno_file):
        super().__init__(vis_processor, text_processor, vis_root, anno_file)

    def __getitem__(self, index):
        ann = self.samples[index]

        image = self.vis_processor(self._read_image(ann, image_key="image_path"))
        question = self.text_processor(ann["question"])

        answer_weight = {}
        for answer in ann["direct_answers"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["direct_answers"])
            else:
                answer_weight[answer] = 1 / len(ann["direct_answers"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        instruction = random.choice(self.VQGA_INSTRUCTIONS)
        prompt = random.choice(self.VQGA_PROMPTS)
        question = instruction + prompt.format(q=question)

        raw_sample = {"img_path": ann["image_path"], "question": ann["question"], "gt_answer": answers,
                      "gt_answer_weight": weights, "image_id": ann["image_id"], "instruction": question}
        if "question_id" in ann:
            raw_sample["question_id"] = ann["question_id"]
        input_sample = {
            "image": image,
            "prompt": question
        }
        return input_sample, raw_sample

    def collater(self, samples):
        image_list, prompt_list, raw_sample_list = [], [], []
        for input_sample, raw_sample in samples:
            raw_sample_list.append(raw_sample)
            image_list.append(input_sample["image"])
            prompt_list.append(input_sample["prompt"])

        return {
            "image": torch.stack(image_list, dim=0),
            "prompt": prompt_list,
            "raw_samples": raw_sample_list
        }


class OKVQA_VQGA_EvalDataset(A_OKVQA_VQGA_EvalDataset):
    def __getitem__(self, index):
        ann = self.samples[index]

        image = self.vis_processor(self._read_image(ann))
        question = self.text_processor(ann["question"])

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        instruction = random.choice(self.VQGA_INSTRUCTIONS)
        prompt = random.choice(self.VQGA_PROMPTS)
        question = instruction + prompt.format(q=question)

        image_id = int(ann["image"].split("_")[-1][:-4])

        raw_sample = {"img_path": ann["image"], "question": ann["question"], "gt_answer": answers,
                      "gt_answer_weight": weights, "image_id": image_id, "instruction": question}
        if "question_id" in ann:
            raw_sample["question_id"] = ann["question_id"]
        input_sample = {
            "image": image,
            "prompt": question
        }
        return input_sample, raw_sample


class VQAv2_VQGA_EvalDataset(OKVQA_VQGA_EvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, anno_file):
        super(VQAv2_VQGA_EvalDataset, self).__init__(vis_processor, text_processor, vis_root, anno_file)
        sample_ratio = 20
        self.samples = self.samples[::sample_ratio]


class COCO2017_Eval_Dataset(A_OKVQA_VQGA_EvalDataset):

    def __init__(self, vis_processor, text_processor, vis_root, anno_file, image_ids_file, filter):
        super().__init__(vis_processor, text_processor, vis_root, anno_file)
        image_ids = json.load(open(image_ids_file))
        _filter_image_ids = {k: v for k, v in image_ids.items() if k in filter}
        filter_image_ids = []
        for ids in _filter_image_ids.values():
            filter_image_ids.extend(ids)
        self.filter_image_ids = set(filter_image_ids)
        filtered_samples = []

        for sample in self.samples:
            if sample["image_id"] not in filter_image_ids:
                filtered_samples.append(sample)
        self.samples = filtered_samples

    def __getitem__(self, index):
        ann = self.samples[index]

        image = self.vis_processor(self._read_image(ann))

        instruction = random.choice(self.VQGA_INSTRUCTIONS)
        # question = self.text_processor(instruction)
        question = instruction
        image_path = ann["image"]
        image_id = int(image_path.split("/")[-1][:-4])

        raw_sample = {"image": ann["image"], "instruction": question, "image_id": image_id}

        input_sample = {
            "image": image,
            "prompt": question
        }
        return input_sample, raw_sample


class Object365_Eval_Dataset(A_OKVQA_VQGA_EvalDataset):

    def __init__(self, vis_processor, text_processor, vis_root, anno_file, image_ids_file, filter):
        super().__init__(vis_processor, text_processor, vis_root, anno_file)
        image_ids = json.load(open(image_ids_file))
        _filter_image_ids = {k: v for k, v in image_ids.items() if k in filter}
        filter_image_ids = []
        for ids in _filter_image_ids.values():
            filter_image_ids.extend(ids)
        self.filter_image_ids = set(filter_image_ids)
        filtered_samples = []

        for sample in self.samples:
            if sample["image_id"] not in filter_image_ids:
                filtered_samples.append(sample)
        self.samples = filtered_samples

    def __getitem__(self, index):
        ann = self.samples[index]

        image = self.vis_processor(self._read_image(ann))

        instruction = random.choice(self.VQGA_INSTRUCTIONS)
        # question = self.text_processor(instruction)
        question = instruction
        image_path = ann["image"]
        # image: train/patch38/objects365_v2_01805764.jpg
        image_id = int(image_path.split("/")[-1].split('_')[-1][:-4])

        raw_sample = {"image": ann["image"], "instruction": question, "image_id": image_id}

        input_sample = {
            "image": image,
            "prompt": question
        }
        return input_sample, raw_sample


class COCO2017_JiaHui_Eval_Dataset(A_OKVQA_VQGA_EvalDataset):
    PROMPTS = (
        "Question: {q} Short answer:",
    )

    def __getitem__(self, index):
        ann = self.samples[index]
        ann = ann.copy()
        image_path = ann["image"].split("COCO2017/")[-1]
        ann["image"] = image_path

        image = self.vis_processor(self._read_image(ann))

        prompt = random.choice(self.PROMPTS)
        question = self.text_processor(ann["question"])

        image_path = ann["image"]
        image_id = int(image_path.split("/")[-1][:-4])

        raw_sample = {"image": ann["image"], "image_id": image_id, "question": ann["question"]}

        input_sample = {
            "image": image,
            "prompt": prompt.format(q=question)
        }
        return input_sample, raw_sample


class LlavaEvalDataset(BaseDataset):
    VQGA_INSTRUCTIONS = {
        "complex_reasoning_77k.json": (
            "Based on the given image, generate an in-depth reasoning question and then answer it.",),
        "conversation_58k.json": (
            "Generate a question based on the content of the given image and then answer it.",),
        "detail_23k.json": (
            "Generate a question to describe the image content in detail and then answer it.",)
    }

    VQGA_PROMPTS = (
        " Question: {q}",
    )

    def __init__(self, vis_processor, text_processor, vis_root, anno_file):
        super().__init__(vis_processor, text_processor, vis_root, anno_file)

    def get_qa_image(self, ann):
        image = self.vis_processor(self._read_image(ann))
        question = self.text_processor(ann["question"])
        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())
        answer = self.text_processor(random.choices(answers, weights=weights, k=1)[0])
        return {"image": image, "question": question, "answer": answer}

    def __getitem__(self, index):
        ann = self.samples[index]

        process_ann = self.get_qa_image(ann)
        image, question, answer = process_ann["image"], process_ann["question"], process_ann["answer"]
        all_instructions = self.VQGA_INSTRUCTIONS[ann["dataset"]]

        instruction = random.choice(all_instructions)
        prompt = random.choice(self.VQGA_PROMPTS)
        question = instruction + prompt.format(q=question)
        image_path = ann["image"]
        image_id = int(image_path.split("_")[-1][:-4])
        raw_sample = {
            "img_path": image_path,
            "question": ann["question"],
            "gt_answer": answer,
            "image_id": image_id,
            "instruction": question,
        }
        if "question_id" in ann:
            raw_sample["question_id"] = ann["question_id"]
            raw_sample["id"] = ann["question_id"]
        input_sample = {
            "image": image,
            "prompt": question
        }
        return input_sample, raw_sample

    def collater(self, samples):
        image_list, prompt_list, raw_sample_list = [], [], []
        for input_sample, raw_sample in samples:
            raw_sample_list.append(raw_sample)
            image_list.append(input_sample["image"])
            prompt_list.append(input_sample["prompt"])

        return {
            "image": torch.stack(image_list, dim=0),
            "prompt": prompt_list,
            "raw_samples": raw_sample_list
        }
