import numpy as np
import pandas as pd
from collections import namedtuple


def pValues(calibrationAlphas, testAlphas, randomized=False):
    testAlphas = np.array(testAlphas)
    sortedCalAlphas = np.sort(calibrationAlphas)

    leftPositions = np.searchsorted(sortedCalAlphas, testAlphas)

    if randomized:
        rightPositions = np.searchsorted(sortedCalAlphas, testAlphas,
                                         side='right')
        ties = rightPositions - leftPositions + 1  # ties in cal set plus the test alpha itself
        randomizedTies = ties * np.random.uniform(size=len(ties))
        return (len(calibrationAlphas) - rightPositions + randomizedTies) / (
                len(calibrationAlphas) + 1)
    else:
        return (len(calibrationAlphas) - leftPositions + 1) / (
                len(calibrationAlphas) + 1)


def regionPredictor(eps, pAct, pInact):
    actInRegion = pAct > eps
    inactInRegion = pInact > eps
    return (actInRegion, inactInRegion)


def hedgedPrediction(pInact, pAct):
    pointPrediction = pAct > pInact
    conf = 1 - np.minimum(pAct,
                          pInact)  # 1 minus second largest, in this case 1 minus minimum
    cred = np.maximum(pAct, pInact)
    return pointPrediction, conf, cred


def cpConfusionMatrix(p0, p1, testY, epsVals=(
        0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95,
        0.99)):
    corr_act_c = []
    errs_act_c = []
    corr_inact_c = []
    errs_inact_c = []
    empties_act_c = []
    empties_inact_c = []
    uncertain_act_c = []
    uncertain_inact_c = []

    p0 = p0.ravel()
    p1 = p1.ravel()
    testY = testY.ravel()

    activeSubset = testY == 1

    for eps in epsVals:
        a, i = regionPredictor(eps, p1, p0)
        corr_act_c.append(np.sum(a & activeSubset & (~i)))
        errs_act_c.append(np.sum((~a) & activeSubset & i))
        corr_inact_c.append(np.sum(i & (~activeSubset) & (~a)))
        errs_inact_c.append(np.sum(((~i) & (~activeSubset) & a)))
        empties_act_c.append(np.sum(((~a) & (~i) & (activeSubset))))
        empties_inact_c.append(np.sum(((~a) & (~i) & (~activeSubset))))
        uncertain_act_c.append(np.sum((a & i) & (activeSubset)))
        uncertain_inact_c.append(np.sum((a & i) & (~activeSubset)))

    cf_tuple = namedtuple('cf_tuple', ['epsilon',
                                       "Positive_predicted_Positive",
                                       "Positive_predicted_Negative",
                                       "Negative_predicted_Negative",
                                       "Negative_predicted_Positive",
                                       "Positive_predicted_Empty",
                                       "Negative_predicted_Empty",
                                       "Positive_predicted_Uncertain",
                                       "Negative_predicted_Uncertain"])
    return cf_tuple(np.array(epsVals),
                    np.array(corr_act_c),
                    np.array(errs_act_c),
                    np.array(corr_inact_c),
                    np.array(errs_inact_c),
                    np.array(empties_act_c),
                    np.array(empties_inact_c),
                    np.array(uncertain_act_c),
                    np.array(uncertain_inact_c))


def cpConfusionMatrix_df(p0, p1, testY, epsVals=(
        0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95,
        0.99)):
    columns = ['epsilon',
               "Positive predicted Positive",
               "Positive predicted Negative",
               "Negative predicted Negative",
               "Negative predicted Positive",
               "Positive predicted Empty",
               "Negative predicted Empty",
               "Positive predicted Uncertain",
               "Negative predicted Uncertain"]
    cf_df = pd.DataFrame({c: v for c, v in zip(columns,
                                               cpConfusionMatrix(p0, p1, testY,
                                                                 epsVals))})

    return cf_df.set_index('epsilon')


from collections import OrderedDict
from sklearn.metrics import average_precision_score


def precision_at_k(y_true, y_score, k=10):
    """Precision at rank k
    Note that this is not the same as fraction of positives among the top k, when there are ties.
    
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    unique_y = np.unique(y_true)
    pos_label = unique_y[1]

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    k_th_smallest = np.partition(y_score, k - 1)[k - 1]

    up_to_rank_k = np.sum(y_score <= k_th_smallest)

    return np.sum(
        y_true[np.argpartition(y_score, k)][:up_to_rank_k] == 1) / up_to_rank_k


def precision_stats(true_y, pred_y):
    stats = (precision_at_k(true_y, pred_y, k=10),
             precision_at_k(true_y, pred_y, k=25),
             precision_at_k(true_y, pred_y, k=50),
             precision_at_k(true_y, pred_y, k=100),
             precision_at_k(true_y, pred_y, k=200),
             average_precision_score(true_y, pred_y))
    return stats


def cp_statistics(c_p_0, c_p_1, p_0, p_1, y, pics_title_part, sup_title_part):
    c_cf = cpConfusionMatrix_df(c_p_0, c_p_1, y)
    return c_cf.groupby('epsilon').agg('mean')


def _leftOver():
    y = np.where(y == 0, -1, 1)
    precisions = pd.DataFrame(columns=(
        'prec_at_k=10', 'prec_at_k=25', 'prec_at_k=50', 'prec_at_k=100',
        'prec_at_k=200', 'avg_prec'))
    # Rank by lowest p_0 and evaluate precision for positives    
    precisions.loc['Positives lowest p_0'] = precision_stats(y, c_p_0)

    # Rank by lowest p_1 and evaluate precision for negatives
    precisions.loc['Negatives lowest p_1'] = precision_stats(-y, c_p_1)

    # Rank by lowest p_0 and evaluate precision for positives    
    precisions.loc['Positives highest p_1'] = precision_stats(y, -c_p_1)

    # Rank by lowest p_1 and evaluate precision for negatives
    precisions.loc['Negatives highest p_0'] = precision_stats(-y, -c_p_0)

    display(precisions)

    f, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    errors_active = (c_cf['Positive predicted Negative'] + c_cf[
        'Positive predicted Empty']).groupby('epsilon').mean() / np.sum(y == 1)
    errors_inactive = (c_cf['Negative predicted Positive'] + c_cf[
        'Negative predicted Empty']).groupby('epsilon').mean() / np.sum(y != 1)
    validity_delta_active = (errors_active - errors_active.index)
    validity_delta_inactive = (errors_inactive - errors_inactive.index)

    ax.plot((0, 1), (0, 1), 'k--')
    ax.plot(errors_active.index, errors_active, "r-",
            label="Errors for Positives")
    ax.plot(errors_inactive.index, errors_inactive, "g-",
            label="Errors for Negatives")

    ax.set_title('Error rate vs. Significance level')
    ax.set_ylabel('Error rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.set_aspect(1,adjustable='box-forced')
    ax.set_aspect(1, adjustable='datalim')
    ax.legend()

    vd_range = np.max(
        np.r_[np.abs(validity_delta_active), np.abs(validity_delta_inactive)])

    ax2.plot(validity_delta_active.index, validity_delta_active, "r-",
             label="Deviation for Positives")
    ax2.plot(validity_delta_inactive.index, validity_delta_inactive, "g-",
             label="Deviation for Negatives")
    ax2.axhline(linestyle="--", color="k", linewidth=1)
    ax2.set_ylim(-vd_range, vd_range)
    ax2.set_title('Deviation from validity vs. Significance level')

    ax2.set_ylabel('Error rate - Significance level')

    f.suptitle("Validity" + sup_title_part, fontsize=16, y=1.1)

    f.tight_layout()

    f.canvas.draw()

    f.savefig(pics_base_name + pics_title_part + "_validity.png", dpi=300)

    return c_cf, precisions
