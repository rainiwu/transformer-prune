if __name__ == "__main__":
    import prune.models as pm
    import prune.metrics as pr

    pr.finetune_squad(*pm.PRETRAINED["squad"]["gpt2"])
