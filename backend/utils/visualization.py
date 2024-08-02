import matplotlib.pyplot as plt
import seaborn as sns
import shap

# save model plots
def plot_results(y_test, y_test_pred_ridge):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_test_pred_ridge)
    plt.xlabel('Actual EDC_delta13C')
    plt.ylabel('Predicted EDC_delta13C')
    plt.title('Actual vs Predicted EDC_delta13C')
    plt.grid(True)
    plt.savefig('./scatter_plot.png')
    plt.close()


def plot_shap_values(shap_values, X_test_stack):
    shap.summary_plot(shap_values, X_test_stack, plot_type="bar", show=False)
    plt.savefig('./shap_plot.png')
    plt.close()
    shap.summary_plot(shap_values, X_test_stack, show=False)
    plt.savefig('./shap_summary_plot.png')
    plt.close()
