from models.lenet import LeNet
from models.vit import ViTWithTempCNN,ViTWithTempCNN1
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor


def get_network(args):
    if args.model == "lenet3":
        model = LeNet(dataset=args.dataset)
    elif args.model == 'vit':
        config = ViTConfig.from_pretrained(args.model_dir, num_labels=args.num_labels, output_hidden_states=True)
        model = ViTWithTempCNN.from_pretrained(args.model_dir, config=config, args=args, num_classes=args.num_labels)
        # model = ViTWithTempCNN1.from_pretrained(args.model_dir, config=config, args=args, num_classes=args.num_labels)
        model.apply(lambda m: setattr(m, 'depth_mult', 0.25))
        model.apply(lambda m: setattr(m, 'width_mult', 0.25))

    return model