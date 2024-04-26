TOPICAL_CHAT_DATA_CONFIG = {
    "text_column": "turns",
    "summary_column": "target",
    "knowledge_column": "knowledge",
}

COMMONSENSE_DIALOG_DATA_CONFIG = {
    "text_column": "turns",
    "summary_column": "target",
    "knowledge_column": "context",
}

DAILY_DIALOG_DATA_CONFIG = {
    "text_column": "turns",
    "summary_column": "target",
    "knowledge_column": "none",
}

BASELINE_CONFIG = {
    "max_length": 40,
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 0,
    "temperature": 0.7,
    "beam_size": 4,
    "num_return_sequences": 1,
    "write_to_file": "auto",
    "context_augmentation_examples": None, # included to unify results csvs across experiments
    "context_code_attention_bias_value": 1,
    "max_context_examples": 0,
    "cross_attention_bias_value": 1,
    "bias_profile": None,
}

GREEDY_CONFIG = {
    "max_length": 40,
    "do_sample": False,
    "beam_size": 1,
    "num_return_sequences": 1,
    "write_to_file": "auto",
}

DEBUG_CONFIG = {
    "max_predict_samples": 5,
    "write_to_file": '',
    "verbose": True,
    "debug": True,
}

KGD_EXPERIMENT_CONFIGS = {
    "xa_knowledge": {
        "cross_attention_bias_value": 5,
        "bias_profile": "knowledge",
    },
    "xa_dialog": {
        "cross_attention_bias_value": 5,
        "bias_profile": "dialog",
    },
    "qu_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/train_questions.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 10,
    },
    "short_qu_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/short_questions.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 5,
    },
    "single_qu_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/train_questions.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 1,
    },
    "qu_ctxt_aug1": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/train_questions.txt",
        "context_code_attention_bias_value": 1,
        "max_context_examples": 10,
    },
    "pos_sent_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/pos_sents.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 5,
    },
    "single_pos_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/pos_sents.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 1,
    },
    "neg_sent_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/neg_sents.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 5,
    },
    "long_pos_sent_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/train_pos_sents.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 10,
    },
    "long_neg_sent_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/train_neg_sents.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 10,
    },
    # "neu_sent_ctxt_aug5": {
    #     "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/train_neu_sents.txt",
    #     "context_code_attention_bias_value": 5,
    #     "max_context_examples": 10,
    # },
    "hedging_contrast_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/hedging_contrast.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 5,
    },
    "hedging_management_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/hedging_management.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 5,
    },
    "hedging_evasion_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/hedging_evasion.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 5,
    },
    "ambig_qu_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/train_ambig_questions.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 10,
    },
    "ambig_excl_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/train_amibig_exclamations.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 10,
    },
    "excl_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/train_exclamations.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 10,
    },
    "e_words_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/E_words.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 10,
    },
    "d_words_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/D_words.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 10,
    },
    "i_words_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/I_words.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 10,
    },
    "n_words_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/N_words.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 10,
    },
}

CSD_EXPERIMENT_CONFIGS = {
    "xa_knowledge": {
        "cross_attention_bias_value": 5,
        "bias_profile": "knowledge",
    },
    "xa_dialog": {
        "cross_attention_bias_value": 5,
        "bias_profile": "dialog",
    },
    "qu_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Commonsense-Dialogues/CSD/contexts/train_questions.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 10,
    },
    "qu_ctxt_aug1": {
        "context_augmentation_examples": "resources/data/Commonsense-Dialogues/CSD/contexts/train_questions.txt",
        "context_code_attention_bias_value": 1,
        "max_context_examples": 10,
    },
    "pos_sent_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Commonsense-Dialogues/CSD/contexts/pos_sents.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 5,
    },
    "pos_sent_ctxt_aug1": {
        "context_augmentation_examples": "resources/data/Commonsense-Dialogues/CSD/contexts/pos_sents.txt",
        "context_code_attention_bias_value": 1,
        "max_context_examples": 5,
    },
}

DD_EXPERIMENT_CONFIGS = {
    "qu_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/DailyDialog/DD/contexts/train_questions.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 10,
    },
    "qu_ctxt_aug1": {
        "context_augmentation_examples": "resources/data/DailyDialog/DD/contexts/train_questions.txt",
        "context_code_attention_bias_value": 1,
        "max_context_examples": 10,
    },
    "pos_sent_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Daily-Dialog/DD/contexts/pos_sents.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 5,
    },
    "pos_sent_ctxt_aug1": {
        "context_augmentation_examples": "resources/data/Daily-Dialog/DD/contexts/pos_sents.txt",
        "context_code_attention_bias_value": 1,
        "max_context_examples": 5,
    },
}