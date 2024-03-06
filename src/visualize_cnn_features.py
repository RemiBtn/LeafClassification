import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from load_data import get_data_loaders


class CNNNet(nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()

        self.conv_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        self.gelu_1 = nn.GELU()

        self.conv_2 = nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(16)
        self.gelu_2 = nn.GELU()

        self.conv_3 = nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(16)
        self.gelu_3 = nn.GELU()

        self.conv_4 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_4 = nn.BatchNorm2d(32)
        self.gelu_4 = nn.GELU()

        self.conv_5 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_5 = nn.BatchNorm2d(32)
        self.gelu_5 = nn.GELU()

        self.conv_6 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_6 = nn.BatchNorm2d(32)
        self.gelu_6 = nn.GELU()

        self.conv_7 = nn.Conv2d(32, 32, kernel_size=3, bias=False)
        self.bn_7 = nn.BatchNorm2d(32)
        self.gelu_7 = nn.GELU()

        self.mlp = nn.Sequential(
            nn.Linear(128, 128, False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 99),
        )

    def forward(self, image):
        x = self.gelu_1(self.bn_1(self.conv_1(image)))
        x = self.gelu_2(self.bn_2(self.conv_2(x)))
        x = self.gelu_3(self.bn_3(self.conv_3(x)))
        x = self.gelu_4(self.bn_4(self.conv_4(x)))
        x = self.gelu_5(self.bn_5(self.conv_5(x)))
        x = self.gelu_6(self.bn_6(self.conv_6(x)))
        x = self.gelu_7(self.bn_7(self.conv_7(x)))
        x = self.mlp(torch.flatten(x, start_dim=1))
        return x


def get_activation(activation, name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def main():
    model = CNNNet()
    model.load_state_dict(torch.load("../runs/models/feature_maps/feature_maps.pth"))

    _, val_loader, _, species = get_data_loaders(
        use_k_fold=False, include_features=None
    )
    test_set, labels = next(iter(val_loader))

    # define hooks to register feature maps
    activation = {}

    h1 = model.conv_1.register_forward_hook(get_activation(activation, "conv_1"))
    h2 = model.conv_2.register_forward_hook(get_activation(activation, "conv_2"))
    h3 = model.conv_3.register_forward_hook(get_activation(activation, "conv_3"))
    h4 = model.conv_4.register_forward_hook(get_activation(activation, "conv_4"))
    h5 = model.conv_5.register_forward_hook(get_activation(activation, "conv_5"))
    h6 = model.conv_6.register_forward_hook(get_activation(activation, "conv_6"))

    # loop to display image + feature maps
    nos_image = 0

    print("### type an image number between 0 and 197 (inclusive)")
    print("### set a negative image number to exit")
    with torch.no_grad():
        while nos_image >= 0:
            fig_0, axarr_0 = plt.subplots()
            fig_1, axarr_1 = plt.subplots(4, 4)
            fig_2, axarr_2 = plt.subplots(4, 4)
            fig_3, axarr_3 = plt.subplots(4, 4)
            fig_4, axarr_4 = plt.subplots(4, 8)
            fig_5, axarr_5 = plt.subplots(4, 8)
            fig_6, axarr_6 = plt.subplots(4, 8)
            # convert input to appropriate format
            # then compute output (--> forward pass)
            model.eval()
            x = test_set[nos_image]
            x = x.unsqueeze(0)
            out = F.softmax(model(x))
            pred = out.argmax(dim=1, keepdim=True).data.cpu().numpy()[0, 0]
            sco = out[0, pred].data.cpu().numpy()

            # display image
            axarr_0.imshow((test_set.data[nos_image, 0]).cpu(), cmap="gray")
            fig_0.suptitle(
                "img {} (lab={} pred={} sco={:2.3f})".format(
                    nos_image, species[labels[nos_image]], species[pred], sco
                )
            )

            # get featuremaps
            activ_1 = activation["conv_1"].squeeze()
            activ_2 = activation["conv_2"].squeeze()
            activ_3 = activation["conv_3"].squeeze()
            activ_4 = activation["conv_4"].squeeze()
            activ_5 = activation["conv_5"].squeeze()
            activ_6 = activation["conv_6"].squeeze()

            # display feature maps
            for idx in range(activ_1.size(0)):
                axarr_1[divmod(idx, 4)].imshow((activ_1[idx]).cpu(), cmap="gray")
            fig_1.suptitle("Conv_1 feature maps", fontsize=16)

            for idx in range(activ_2.size(0)):
                axarr_2[divmod(idx, 4)].imshow((activ_2[idx]).cpu(), cmap="gray")
            fig_2.suptitle("Conv_2 feature maps", fontsize=16)

            for idx in range(activ_3.size(0)):
                axarr_3[divmod(idx, 4)].imshow((activ_3[idx]).cpu(), cmap="gray")
            fig_3.suptitle("Conv_3 feature maps", fontsize=16)

            for idx in range(activ_4.size(0)):
                axarr_4[divmod(idx, 8)].imshow((activ_4[idx]).cpu(), cmap="gray")
            fig_4.suptitle("Conv_4 feature maps", fontsize=16)

            for idx in range(activ_5.size(0)):
                axarr_5[divmod(idx, 8)].imshow((activ_5[idx]).cpu(), cmap="gray")
            fig_5.suptitle("Conv_5 feature maps", fontsize=16)

            for idx in range(activ_6.size(0)):
                axarr_6[divmod(idx, 8)].imshow((activ_6[idx]).cpu(), cmap="gray")
            fig_6.suptitle("Conv_6 feature maps", fontsize=16)

            fig_0.canvas.draw()
            fig_1.canvas.draw()
            fig_2.canvas.draw()
            fig_3.canvas.draw()
            fig_4.canvas.draw()
            fig_5.canvas.draw()
            fig_6.canvas.draw()

            plt.show()

            nos_image = int(input("Image number ?: "))

    # detach the hooks
    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()
    h5.remove()
    h6.remove()


if __name__ == "__main__":
    main()
