from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# model = resnet50(pretrained=True)
# target_layer = model.layer4[-1]
# input_tensor = # Create an input tensor image for your model..
# # Note: input_tensor can be a batch tensor with several images!

N_LABEL = 14

class DenseNet121(nn.Module):
    """
    The last layer of DenseNet121 was replaced by a Linear with 14 output features, followed by a sigmoid function
    """
    def __init__(self, out_feature):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        in_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(in_features, out_feature),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

model = DenseNet121(N_LABEL)
# load trained model if needed
model.load_state_dict(torch.load("./saved_model/8trained.pth"))
target_layer = model.classifier
# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layer=target_layer)

# If target_category is None, the highest scoring category
# will be used for every image in the batch.
# target_category can also be an integer, or a list of different integers
# for every image in the batch.
target_category = 281

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam)