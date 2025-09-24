import argparse
import os
import json
import pdb
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
import pandas as pd
from constant import POPE_PATH
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np


eval_type_dict = {
    "Perception": ["existence", "count", "position", "color"]
}


class calculate_metrics_mme:
    def divide_chunks(self, l, n=2):
        # looping till length l
        for i in range(0, len(l), n): 
            yield l[i:i + n]
        
        return 

    def parse_pred_ans(self, pred_ans):
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]

            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"

        return pred_label


    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            "yes": 1,
            "no": 0,
            "other": -1,
        }
        
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds) 

        clean_gts = []
        clean_preds = []
        other_num = 0 
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
        

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        return metric_dict


    def process_result(self, results_dir):
        results = {}
        model_score_dict = dict()
        for eval_type, task_name_list in eval_type_dict.items():
           
            scores = 0
            task_score_dict = dict()

            for task_name in task_name_list:

                task_txt = os.path.join(results_dir, task_name + ".txt")
                lines = open(task_txt, 'r').readlines()
                chunk_lines = list(self.divide_chunks(lines)) # one image corresponds to two questions
                
                img_num = len(chunk_lines)
                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts = []
                preds = []

                for img_items in chunk_lines:
                    assert len(img_items) == 2
                    img_correct_num = 0

                    for img_item in img_items:
                        try:
                            img_name, question, gt_ans, pred_ans = img_item.split("\t")
                        except:
                            import pdb; pdb.set_trace()
                        gt_ans = gt_ans.lower()
                        pred_ans = pred_ans.lower()

                        assert gt_ans in ["yes", "no"] # gt can only be yes or no.

                        pred_ans = self.parse_pred_ans(pred_ans)
                        assert pred_ans in ["yes", "no", "other"]

                        gts.append(gt_ans)
                        preds.append(pred_ans)
                        
                        if gt_ans == pred_ans:
                            img_correct_num += 1
                        
                        if pred_ans not in ["yes", "no"]:
                            task_other_ans_num += 1

                    if img_correct_num == 2:
                        acc_plus_correct_num += 1

                # cal TP precision acc, etc.
                metric_dict = self.compute_metric(gts, preds)
                acc_plus = acc_plus_correct_num / img_num
                metric_dict["acc_plus"] = acc_plus
                
                
                for k, v in metric_dict.items():
                    if k in ["acc", "acc_plus"]:
                        task_score += v*100
                
                task_score_dict[task_name] = task_score
                
                scores += task_score

            for task_name, score in task_score_dict.items():
                results.update({task_name: score})
        
        return results


def extract_question_id_label(file_path):
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            result = {'question_id': data.get('question_id'),
                      'label': data.get('label'),
                      'type': data.get('type')}
            results.append(result)
    return results

# NOTE
def extract_meta_answer(text, gt_answer, dataset_type):
    if dataset_type == 'POPE':
        gen_answer = text.lower()
        gen_answer = gen_answer.strip()
        gt_answer = gt_answer['label']
        gt_answer = gt_answer.lower()
        if gt_answer == 'yes':
            if 'yes' in gen_answer:
                return 'yes'
            else:
                return 'no'
        elif gt_answer == 'no':
            if 'no' in gen_answer:
                return 'no'
            else:
                return 'yes'
    elif dataset_type == 'MME':
        return text


def extract_answer_id_label(file_path, dataset_type, gt_answers=None):
    results = []
    with open(file_path, 'r') as file:
        for idx, line in enumerate(file):
            data = json.loads(line)
            model_answer = data.get('text')
            if dataset_type == 'MME':
                if model_answer == '' or model_answer is None:
                    model_answer = 'other'
                else:
                    model_answer = model_answer
            else:
                model_answer = extract_meta_answer(model_answer, gt_answers[idx], dataset_type)
            question_id = data.get('question_id')
            metadata = data.get('metadata')
            result = {'question_id': question_id, 'label': model_answer, 'metadata': metadata}
            results.append(result)
    return results


def compute_acc(gp, predict):
    assert len(gp) == len(predict)
    cnt = 0
    cnt_object = 0
    cnt_attribute = 0
    cnt_relation = 0
    o, a, r = 0, 0, 0
    for idx in range(len(gp)):
        if gp[idx]['type'] == 'Object':
            o += 1
        elif gp[idx]['type'] == 'Attribute':
            a += 1
        elif gp[idx]['type'] == 'Relation':
            r += 1
        
        if gp[idx]['label'] == predict[idx]['label']:
            cnt += 1
            if gp[idx]['type'] == 'Object':
                cnt_object += 1
            elif gp[idx]['type'] == 'Attribute':
                cnt_attribute += 1
            elif gp[idx]['type'] == 'Relation':
                cnt_relation += 1
    return cnt / len(gp), cnt_object / o, cnt_attribute / a, cnt_relation / r


def compute_for_pope(gp, predict):
    # gp[idx]['label'] can only be 'yes' or 'no'
    assert len(gp) == len(predict)

    tp = fp = tn = fn = 0

    for idx in range(len(gp)):
        true_label = gp[idx]['label']
        predicted_label = predict[idx]['label']

        if true_label == 'yes':
            if predicted_label == 'yes':
                tp += 1  # True Positive
            else:
                fn += 1  # False Negative
        else:  # true_label == 'no'
            if predicted_label == 'no':
                tn += 1  # True Negative
            else:
                fp += 1  # False Positive

    # Calculate metrics
    accuracy = (tp + tn) / len(gp) if len(gp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1score


def calculate_MME(answers):
    destination_path = '/home/bingkui/HallucinationCD/dataset/MME/temporary'
    import shutil
    shutil.copytree('/home/bingkui/HallucinationCD/dataset/MME/result_template',
                destination_path,
                dirs_exist_ok=True)
    def append_to_line(file_path, line_number, append_content):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        if line_number >= len(lines) or line_number < 0:
            raise ValueError("行号超出范围")

        lines[line_number] = lines[line_number].strip('\n') + '\t' + append_content + '\n'

        with open(file_path, 'w') as file:
            file.writelines(lines)
    for idx, answer in enumerate(answers):
        revised_answer = answer['label'].replace("\t", "").replace("\r", "").replace("\n", "").strip()
        if revised_answer == '' or revised_answer is None:
            revised_answer = 'other'
        append_to_line(file_path=os.path.join(destination_path, answer['metadata'] + '.txt'),
                       line_number=answer['question_id'],
                       append_content=revised_answer)
    cal = calculate_metrics_mme()

    results_dir = destination_path
    results = cal.process_result(results_dir)
    return results

def main(args):
    dataset_type = args.dataset

    accuracies = {}
    results_pope = []
    cd_types = ['original', 'layercd', 'VCD']
    model_types = ['LLaVA', 'Molmo', 'Cambrian']
    for cd_type in cd_types:
        for model_type in model_types:
            if dataset_type == 'POPE':
                # rewrite
                for POPE_sampling_type in ['coco', 'aokvqa', 'gqa']:
                    for POPE_type in ['popular', 'random', 'adversarial']:
                        accuracy_seed, precision_seed, recall_seed, f1score_seed = [], [], [], []
                        for seed in range(1, 6):
                            question_file = POPE_PATH.format(POPE_sampling_type, POPE_sampling_type, POPE_type)
                            gt_answers = extract_question_id_label(question_file)
                            answer_file = args.answers_folder.format(cd_type, model_type, dataset_type) + '{}_{}_seed_{}.json'.format(POPE_sampling_type, POPE_type, seed)
                            answers = extract_answer_id_label(answer_file, dataset_type, gt_answers)
                            if len(gt_answers) != len(answers):
                                print('broken file: ', answer_file)
                            accuracy, precision, recall, f1score = compute_for_pope(gt_answers, answers)
                            accuracy_seed.append(accuracy)
                            precision_seed.append(precision)
                            recall_seed.append(recall)
                            f1score_seed.append(f1score)

                        mean_accuracy = np.mean(accuracy_seed)
                        std_accuracy = np.std(accuracy_seed)
                        mean_precision = np.mean(precision_seed)
                        std_precision = np.std(precision_seed)
                        mean_recall = np.mean(recall_seed)
                        std_recall = np.std(recall_seed)
                        mean_f1score = np.mean(f1score_seed)
                        std_f1score = np.std(f1score_seed)

                        results_pope.append([f'{model_type}_{cd_type}_{POPE_sampling_type}_{POPE_type}',
                                            [float(mean_accuracy), float(std_accuracy)],
                                            [float(mean_precision), float(std_precision)],
                                            [float(mean_recall), float(std_recall)],
                                            [float(mean_f1score), float(std_f1score)]])   
            elif dataset_type == 'MME':
                mme_types_accuracies = {'existence': [], 'count': [], 'position': [], 'color': []}
                for seed in range(1, 6):
                    answer_file = args.answers_folder.format(cd_type, model_type, dataset_type) + 'MME_seed_{}.json'.format(seed)
                    answers = extract_answer_id_label(answer_file, dataset_type)
                    results = calculate_MME(answers)
                    for mme_type, score in results.items():
                        mme_types_accuracies[mme_type].append(score)
                for mme_type, score in mme_types_accuracies.items():
                    mean_score = np.mean(score)
                    std_score = np.std(score)
                    accuracies.update({f'{model_type}_{cd_type}_{mme_type}': [float(mean_score), float(std_score)]})
    

    if dataset_type == 'POPE':
        df = pd.DataFrame(results_pope, columns=['type', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    else:
        df = pd.DataFrame(list(accuracies.items()), columns=['type', 'Accuracy'])
    save_folder = '/home/bingkui/HallucinationCD/answer'
    if dataset_type == 'POPE':
        file_name = os.path.join(save_folder, f"accuracies_{dataset_type}.csv")
    elif dataset_type == 'MME':
        file_name = os.path.join(save_folder, f"accuracies_{dataset_type}.csv")

    df.to_csv(file_name, index=False)
    print(f"Accuracies saved to {file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # cd_type model_type /home/bingkui/HallucinationCD/answer/layercd/LLaVA/MCQ/beam_3_alpha_1_beta_0.1_noise_step_500
    parser.add_argument("--answers-folder", type=str, default="/home/bingkui/HallucinationCD/answer/{}/{}/{}/temperature_1.0_topk_None_topp_1_alpha_1_beta_0.1_noise_step_500/")
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    main(args)

# python util/compute_results.py --dataset=POPE
# python util/compute_results.py --dataset=MME