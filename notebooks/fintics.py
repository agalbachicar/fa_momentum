import datetime
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import time

from sklearn import datasets
from sklearn.metrics import roc_curve, auc, classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection._split import _BaseKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, BaggingClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from scipy import interp
from scipy.stats import norm

from itertools import cycle

from statsmodels.tsa.stattools import adfuller

def plot_mtum(df):
  '''
  Crea una figura con dos graficos en columna.
  El grafico de arriba imprime la evolucion del precio de cierre, maximo y minimo de forma diaria.
  El grafico de abajo imprime la evolucion del volumen operado en el dia.
  
  @param df Es el data frame de pandas de donde se extraen los valores.
            Espera que tenga cuatro series completas: 'Close','High', 'Low' y 'Date'.
     
  '''
  fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20,10))
  
  df.plot(kind='line',y='Close', x='Date', color='blue', ax=axes[0])
  df.plot(kind='line',y='High', x='Date', color='green', ax=axes[0])
  df.plot(kind='line',y='Low', x='Date', color='red', ax=axes[0])
  df.plot(kind='line',y='Open', x='Date', color='orange', ax=axes[0])
  plt.title('MTUM prices')

  df.plot(kind='line',y='Volume', x='Date', color='blue', ax=axes[1])
  plt.title('MTUM volume')

  plt.show()

def tick_bars(df, price_column, m):
  '''
  compute tick bars

  # args
      df: pd.DataFrame()
      column: name for price data
      m: int(), threshold value for ticks
  # returns
      idx: list of indices
  '''
  t = df[price_column]
  ts = 0
  idx = []
  for i, x in enumerate(t):
    ts += 1
    if ts >= m:
      idx.append(i)
      ts = 0
      continue
  return idx

def tick_bar_df(df, price_column, m):
  '''
  Filtra `df` por los tick_bars 
  '''
  idx = tick_bars(df, price_column, m)
  return df.iloc[idx].drop_duplicates()

def volume_bars(df, volume_column, m):
  '''
  compute volume bars

  # args
      df: pd.DataFrame()
      volume_column: name for volume data
      m: int(), threshold value for volume
  # returns
      idx: list of indices
  '''
  t = df[volume_column]
  ts = 0
  idx = []
  for i, x in enumerate(t):
    ts += x
    if ts >= m:
      idx.append(i)
      ts = 0
      continue
  return idx

def volume_bar_df(df, volume_column, m):
  idx = volume_bars(df, volume_column, m)
  return df.iloc[idx].drop_duplicates()

def create_dollar_volume_series(df, price_col, volume_col):
  return df[price_col] * df[volume_col]

def dollar_bars(df, dv_column, m):
  '''
  compute dollar bars

  # args
      df: pd.DataFrame()
      dv_column: name for dollar volume data
      m: int(), threshold value for dollars
  # returns
      idx: list of indices
  '''
  t = df[dv_column]
  ts = 0
  idx = []
  for i, x in enumerate(t):
    ts += x
    if ts >= m:
      idx.append(i)
      ts = 0
      continue
  return idx

def dollar_bar_df(df, dv_column, m):
  idx = dollar_bars(df, dv_column, m)
  return df.iloc[idx].drop_duplicates()

def tick_direction(prices):
  '''
  Computa un vector de ticks {1, -1} cuyo signo indica el valor
  del retorno entre dos muestras consecutivas.
  El valor inicial es el mismo que el primero computado.
  El vector de retorno tiene el mismo tamaño que @p prices.
  
  @param prices Es un vector de precios a diferenciar y obtener el signo del retorno.
  @return b_t, un vector de tick imbalance bars.
  '''
  tick_directions = prices.diff()
  tick_directions[0] = tick_directions[1]
  tick_directions = tick_directions.transform(lambda x: np.sign(x))
  return tick_directions

def signed_volume(tick_directions, volumes):
  '''
  Computa una serie de volumenes signados segun el computo de ticks.
  
  @param tick_directions La serie con el signo de del retorno.
  @param volumes La serie de volumenes para cada sample temporal de retorno.
  @return Una serie de volumenes signados, o bien el producto elemento a elemento de
          @p tick_directions con @p volumes.
  '''
  return tick_directions.multiply(volumes)

def exponential_weighted_moving_average(arr_in, window):
  '''
  @see https://stackoverflow.com/a/51392341
  
  Exponentialy weighted moving average specified by a decay ``window``
  assuming infinite history via the recursive form:

      (2) (i)  y[0] = x[0]; and
          (ii) y[t] = a*x[t] + (1-a)*y[t-1] for t>0.

  This method is less accurate that ``_ewma`` but
  much faster:

      In [1]: import numpy as np, bars
         ...: arr = np.random.random(100000)
         ...: %timeit bars._ewma(arr, 10)
         ...: %timeit bars._ewma_infinite_hist(arr, 10)
      3.74 ms ± 60.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
      262 µs ± 1.54 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

  Parameters
  ----------
  arr_in : np.ndarray, float64
      A single dimenisional numpy array
  window : int64
      The decay window, or 'span'

  Returns
  -------
  np.ndarray
      The EWMA vector, same length / shape as ``arr_in``

  Examples
  --------
  >>> import pandas as pd
  >>> a = np.arange(5, dtype=float)
  >>> exp = pd.DataFrame(a).ewm(span=10, adjust=False).mean()
  >>> np.array_equal(_ewma_infinite_hist(a, 10), exp.values.ravel())
  True
  '''
  n = arr_in.shape[0]
  ewma = np.empty(n, dtype=float)
  alpha = 2 / float(window + 1)
  ewma[0] = arr_in[0]
  for i in range(1, n):
    ewma[i] = arr_in[i] * alpha + ewma[i-1] * (1 - alpha)
  return ewma

def compute_initial_e_v(signed_volumes):
  '''
  Computa el valor absoluto de la media de los volumenes signados.
  Sirve como estimacion del valor inicial de Φ_T para toda la serie de volumenes.
  '''
  return abs(signed_volumes.mean())

def compute_tick_imbalance(signed_volumes, e_t_0, abs_e_v_0):
  '''
  @param signed_volumes Serie de volumenes signados.
  @param e_t_0 El valor inicial de la $E(T)$
  @param abs_e_v_0 El valor absoluto del valor medio (hint) de $Φ_T$.
  @return Una tupla {Ts, abs_thetas, thresholds, i_s} donde:
      Ts: es un vector con los valores de $T$ que se tomaron como largo de ventana de EWMA.
      abs_thetas: es un vector que indica los valores de Φ_T para cada valor de volumen.
      thresholds: es un vector que indica el valor the umbrales que se como para cada valor de volumen.
      i_s: es un vector con los valores de los indices que referencia al vector de volumen con un cambio de tick.
  '''
  Ts, i_s = [], []
  
  # Valores de la iteracion
  # i_prev: valor de indice previo donde se fijo $T$.
  # e_t: $E(T)$ iteracion a iteracion.
  # abs_e_v: $|Φ_T|$ iteracion a iteracion.
  i_prev, e_t, abs_e_v  = 0, e_t_0, abs_e_v_0
  
  n = signed_volumes.shape[0]
  signed_volumes_val = signed_volumes.values.astype(np.float64)
  abs_thetas, thresholds = np.zeros(n), np.zeros(n)
  abs_thetas[0], cur_theta = np.abs(signed_volumes_val[0]), signed_volumes_val[0]
  for i in range(1, n):
    cur_theta += signed_volumes_val[i]
    abs_theta = np.abs(cur_theta)
    abs_thetas[i] = abs_theta
    
    threshold = e_t * abs_e_v
    thresholds[i] = threshold
    
    if abs_theta >= threshold:
      cur_theta = 0
      Ts.append(np.float64(i - i_prev))
      i_s.append(i)
      i_prev = i
      e_t = exponential_weighted_moving_average(np.array(Ts), window=np.int64(len(Ts)))[-1]
      abs_e_v = np.abs(exponential_weighted_moving_average(signed_volumes_val[:i], window=np.int64(e_t_0 * 3))[-1] ) # window of 3 bars
  return Ts, abs_thetas, thresholds, i_s

def compute_ewma(prices, window_size):
  '''
  Computes the EWMA of a price series with a certain window size.
  
  @param prices A pandas series.
  @param window_size EWMA window size.
  @return The EWMA with `window_size` window size of `prices`.
  '''
  return prices.ewm(window_size).mean()

def get_up_cross(fast_ewma, slow_ewma):
  '''
  Computes the fast EWMA serie cross over the slow EWMA serie.
  
  @param fast_ewma A fast EWMA pandas series.
  @param slow_ewma A slow EWMA pandas series.
  @return A filtered version of `fast_ewma` that indicates when the buy trend starts.
  '''
  crit1 = fast_ewma.shift(1) < slow_ewma.shift(1)
  crit2 = fast_ewma > slow_ewma
  return fast_ewma[(crit1) & (crit2)]

def get_down_cross(fast_ewma, slow_ewma):
  '''
  Computes the slow EWMA serie cross over the fast EWMA serie.
  
  @param fast_ewma A fast EWMA pandas series.
  @param slow_ewma A slow EWMA pandas series.
  @return A filtered version of `fast_ewma` that indicates when the sell trend starts.
  '''
  crit1 = fast_ewma.shift(1) > slow_ewma.shift(1)
  crit2 = fast_ewma < slow_ewma
  return fast_ewma[(crit1) & (crit2)]

def create_bet_signal_fast_slow_ewma(df, price_column, fast_window_size, slow_window_size):
  '''
  Computes the buy / sell events based on the 50-200 EWMA cross.
  
  Appends three series to `df`:
  1- FastEWMA : the fast EWMA computed with `fast_window_size`.
  2- SlowEWMA : the fast EWMA computed with `slow_window_size`.
  3- BetEWMA : an integer series with {1, 0, -1} values meaning {Buy, Do nothing, Sell}.
  
  @param df A pandas data frame to extract the price series from.
  @param price_column A string telling the name of the price series.
  @param fast_window_size The fast EWMA window size.
  @param slow_window_size The slow EWMA window size.
  @return `df` with the appended columns.
  '''
  fast_ewma = compute_ewma(df[price_column], fast_window_size)
  slow_ewma = compute_ewma(df[price_column], slow_window_size)
  buy_bet = get_up_cross(fast_ewma, slow_ewma)
  sell_bet = get_down_cross(fast_ewma, slow_ewma)
  
  df['FastEWMA'] = fast_ewma
  df['SlowEWMA'] = slow_ewma
  df['BetEWMA'] = 0
  df.BetEWMA.iloc[buy_bet.index] = 1
  df.BetEWMA.iloc[sell_bet.index] = -1
  return df

def plot_ewma_bet_signals(df):
  f, ax = plt.subplots(figsize=(20,10))

  df.plot(ax=ax, alpha=.5, y='Close', x='Date', color='blue')
  df.plot(ax=ax, alpha=.5, y='FastEWMA', x='Date', color='yellow')
  df.plot(ax=ax, alpha=.5, y='SlowEWMA', x='Date', color='brown')
  df.Close.loc[df.BetEWMA == 1].plot(ax=ax, ls='', marker='^', markersize=7, alpha=0.75, label='Buy', color='green')
  df.Close.loc[df.BetEWMA == -1].plot(ax=ax, ls='', marker='v', markersize=7, alpha=0.75, label='Sell', color='red')
  ax.grid()
  ax.legend()

def getDailyVol(close,span0=100):
    '''
    Computes the daily volatility of price returns.
    It takes a closing price series, applies a diff sample to sample
    (assumes each sample is the closing price), computes an EWM with 
    `span0` samples and then the standard deviation of it.
    
    @param[in] close A series of prices where each value is the closing price of an asset.
    @param[in] span0 The sample size of the EWM.
    @return A pandas series of daily return volatility.
    '''
    df0=close.diff()
    df0=df0 - 1
    df0[0]=0
    df0=df0.ewm(span=100).std().rename('dailyVol')
    df0[0]=df0[1]
    return df0

def getTEvents(close, h):
    '''
    Computes a pandas series of indices of `df[price_col]` that are the output
    of a CUSUM positive and negative filter. The threshold of the filter is `h`.
    
    @param[in] close A series of prices where each value is the closing price of an asset.
    @param[in] h CUSUM filter threshold.
    @return A pandas index series that mark where the CUSUM filter flagged either positive
    and negative cumulative sums that are bigger than `h`.
    '''
    tEvents, sPos, sNeg = [], 0, 0
    diff = close.diff()
    diff[0] = 0
    for i in diff.index:
      sPos, sNeg = max(0, sPos+diff.loc[i]), min(0, sNeg + diff.loc[i])
      if sNeg < -h:
        sNeg = 0
        tEvents.append(i)
      if sPos > h:
        sPos = 0
        tEvents.append(i)
    return pd.Int64Index(tEvents)
    

def addVerticalBarrier(tEvents, close, numDays=1):
    '''
    Returns a filtered pandas series of prices coming from `close` that
    belong to the offset price in `numDays` of `tEvents` prices.
    
    @param[in] tEvents A pandas index series that match the same type of `close`'s index.
    @param[in] close A series of prices where each value is the closing price of an asset.
    @param[in] numDays A delta in samples to apply to all a vertical barrier.
    @return A pandas series of prices. 
    '''
    t1=close.index.searchsorted(tEvents + numDays)
    t1=t1[t1<close.shape[0]]
    t1=(pd.Series(close.index[t1],index=tEvents[:t1.shape[0]]))
    return t1

def applyPtSlOnT1(close,events,ptSl,molecule):
    '''
    Apply stop loss/profit taking, if it takes place before t1 (end of event)
    
    @param[in] close A pandas series of prices.
    @param[in] events A pandas dataframe, with columns:
      - `t1`:  the timestamp of vertical barries. When the value is np.nan, there will not be a vertical barrier.
      - `trgt`: the unit width of the horizontal barriers.
    @param[in] ptSl A list of two non-negative float values:
      - `ptSl[0]`: the factor that multiplies `trgt` to set the width of the upper barrier. If 0, there will not be an upper barrier.
      - `ptSl[1]`: the factor that multiplies `trgt` to set the width of the lower barrier. If 0, there will not be an lower barrier.
    @param[in] molecule  A list with the subset of event indices that will be processed by a single thread.
    '''
    events_=events.loc[molecule]
    out=events_[['t1']].copy(deep=True)
    if ptSl[0]>0: pt=ptSl[0]*events_['trgt']
    else: pt=pd.Series(index=events.index) # NaNs
    if ptSl[1]>0: sl=-ptSl[1]*events_['trgt']
    else: sl=pd.Series(index=events.index) # NaNs
    for loc,t1 in events_['t1'].fillna(close.index[-1]).iteritems():
      loc = int(loc)
      t1 = int(t1)
      df0=close[loc:t1] # path prices
      df0=(df0/close[loc]-1)*events_.at[loc,'side'] # path returns
      out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss
      out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking
    return out

def getEvents(close, tEvents, ptSl, trgt, minRet, t1=False, side=None):
    #1) get target
    trgt=trgt.loc[tEvents]
    trgt=trgt[trgt>minRet] # minRet
    #2) get t1 (max holding period)
    if t1 is False:t1=pd.Series(pd.NaT, index=tEvents)
    #3) form events object, apply stop loss on t1
    if side is None:side_,ptSl_=pd.Series(1.,index=trgt.index), [ptSl[0],ptSl[0]]
    else: side_,ptSl_=side.loc[trgt.index],ptSl[:2]
    events=(pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1)
            .dropna(subset=['trgt']))
    df0=applyPtSlOnT1(close, events, ptSl_, events.index)
    events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan
    if side is None:events=events.drop('side',axis=1)
    return events
    
def getBins(events, close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if 'side' in events_:out['ret']*=events_['side'] # meta-labeling
    out['bin']=np.sign(out['ret'])
    if 'side' in events_:out.loc[out['ret']<=0,'bin']=0 # meta-labeling
    return out

def getBinsNew(events, close, t1=None):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    -t1 is original vertical barrier series
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if 'side' in events_:out['ret']*=events_['side'] # meta-labeling
    out['bin']=np.sign(out['ret'])
    
    if 'side' not in events_:
      # only applies when not meta-labeling
      # to update bin to 0 when vertical barrier is touched, we need the original
      # vertical barrier series since the events['t1'] is the time of first 
      # touch of any barrier and not the vertical barrier specifically. 
      # The index of the intersection of the vertical barrier values and the 
      # events['t1'] values indicate which bin labels needs to be turned to 0
      vtouch_first_idx = events[events['t1'].isin(t1.values)].index
      out.loc[vtouch_first_idx, 'bin'] = 0.
    
    if 'side' in events_:out.loc[out['ret']<=0,'bin']=0 # meta-labeling
    return out

def getRandomForest(n_estimator=150, oob_score=False, max_samples=None):
  return RandomForestClassifier(max_depth=2, n_estimators=n_estimator, criterion='entropy', class_weight='balanced_subsample', random_state=RANDOM_STATE, oob_score=oob_score, max_samples=max_samples)

def plotROC(y_test, y_pred_rf):
  fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

  plt.figure(1)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.plot(fpr_rf, tpr_rf, label='RF')
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.show()

def train_test_samples(events, labels, test_size, binarize = False):
  X = events_side.dropna().values.reshape(-1,1)
  y = labels.bin.values
  if binarize: y = label_binarize(y, [-1, 0, 1])
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
  return X, y, X_train, X_test, y_train, y_test


def getSignal(events, stepSize, prob, pred, numClasses, **kargs):
  if prob.shape[0] == 0: return pd.Series()
  signal0 = (prob - 1. / numClasses) / (prob * (1. - prob)) ** 0.5
  signal0 = pred * (2. * norm.cdf(signal0) - 1.)
  return signal0
  # if 'side' in events: signal0*=events.loc[signal0.index, 'side']
  # df0 = signal0.to_frame('signal').join(events[['t1']], how='left')
  # df0 = avgActiveSignals(df0)
  # signal1 = discreteSignal(signal0=pd.Series(signal0), stepSize=stepSize)
  # return signal1

def avgActiveSignals(signals):
  tPnts = set(signals['t1'].dropna().values)
  tPnts = tPnts.union(signals.index.values)
  tPnts = list(tPnts); tPnts.sort()
  out = mpAvgActiveSignals(signals, ('molecule', tPnts))
  return out

def mpAvgActiveSignals(signals, molecule):
  out = pd.Series()
  for loc in molecule:
    df0 = (signals.index.values <= loc) & ((loc < signals['t1']) | pd.isnull(signals['t1']))
    act = signals[df0].index
    if len(act) > 0: out[loc] = signals.loc[act, 'signal'].mean()
    else: out[loc] = 0
  return out

def discreteSignal(signal0, stepSize):
  signal1 = (signal0 / stepSize).round() * stepSize
  signal1[signal1 > 1] = 1
  signal1[signal1 < 1] = -1
  return signal1

def getOptimizedRandomForest(max_samples=None, oob_score=False):
    n_estimators = [50, 100, 150, 300]
    max_depth = [2, 3, 4,]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [2, 3, 5,]
    n_iter = 100

    grid_params = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    if RUN_RANDOM_FOREST_OPTIMIZATION:
      random_forest_classifier = RandomForestClassifier(random_state=RANDOM_STATE, oob_score=oob_score, max_samples=max_samples)
      return RandomizedSearchCV(estimator = random_forest_classifier, param_distributions=grid_params, n_iter=n_iter, cv=CV, verbose=2, random_state=RANDOM_STATE**2, n_jobs=-1)
    return getRandomForest(n_estimator=150, oob_score=oob_score, max_samples=max_samples)

def getWeights(d,size):
  # thres>0 drops insignificant weights
  w=[1.]
  for k in range(1,size):
    w_=-w[-1]/k*(d-k+1)
    w.append(w_)
  w=np.array(w[::-1]).reshape(-1,1)
  return w

def plotWeights(dRange,nPlots,size):
  w=pd.DataFrame()
  for d in np.linspace(dRange[0],dRange[1],nPlots):
    w_=getWeights(d,size=size)
    w_=pd.DataFrame(w_,index=range(w_.shape[0])[::-1],columns=[d])
    w=w.join(w_,how='outer')
  ax=w.plot()
  ax.legend(loc='upper left');mpl.show()
  return

def plotWeights_FFD(dRange,nPlots,thres):
  w=pd.DataFrame()
  for d in np.linspace(dRange[0],dRange[1],nPlots):
    w_=getWeights_FFD(d,thres=thres)
    w_=pd.DataFrame(w_,index=range(w_.shape[0])[::-1],columns=[d])
    w=w.join(w_,how='outer')
  ax=w.plot()
  ax.legend(loc='upper left');mpl.show()
  return

def getWeights_FFD(d, thres):
  w, k = [1.0], 1
  while True:
    w_ = -w[-1] / k * (d - k + 1)
    if abs(w_) < thres:
        break
    w.append(w_)
    k += 1
  return np.array(w[::-1]).reshape(-1, 1)


def fracDiff_FFD(series, d, thres=1e-5):
  # Constant width window (new solution)
  #Note 1: thres determines the cut-off weight for the window
  #Note 2: d can be any positive fractional, not necessarily bounded [0,1]
  
  #1) Compute weights for the longest series
  w =  getWeights_FFD(d, thres)
  width, df = len(w) - 1, {}
  
  #2) Apply weights to values
  for name in series.columns:
    seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series(index=series.index)
    for iloc1 in range(width, seriesF.shape[0]):
      loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
      if not np.isfinite(series.loc[loc1, name]):
        continue # exclude NAs
      df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
    df[name] = df_.copy(deep=True)
  df = pd.concat(df, axis=1)
  return df

def plotMinFFD(close, threshold, testMinFFDFileName = None):
  out= pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf','corr'])
  df0= close
  for d in np.linspace(0,1,21):
    df1=np.log(df0).resample('1D').last().dropna() # downcast to daily obs. Dropped NAs
    df2=fracDiff_FFD(df1, d, thres=threshold).dropna()
    corr=np.corrcoef(df1.loc[df2.index,'Close'],df2['Close'])[0,1]
    df2=adfuller(df2['Close'],maxlag=1,regression='c',autolag=None)
    out.loc[d]=list(df2[:4])+[df2[4]['5%']]+[corr] # with critical value
  if testMinFFDFileName is not None: out.to_csv(testMinFFDFileName)
  out[['adfStat','corr']].plot(secondary_y='adfStat')
  mpl.pyplot.axhline(out['95% conf'].mean(),linewidth=1,color='r',linestyle='dotted')
  #mpl.savefig(path+instName+'_testMinFFD.png')
  return out

def evaluate(X,y,clf):
  # The random forest model by itself
  y_pred_rf = clf.predict_proba(X)[:, 1]
  y_pred = clf.predict(X)
  fpr_rf, tpr_rf, _ = roc_curve(y, y_pred_rf)
  print(classification_report(y, y_pred))

  plt.figure(figsize=(9,6))
  plt.plot([0, 1], [0, 1], 'k--')
  plt.plot(fpr_rf, tpr_rf, label='clf')
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.show()
    
    
def evaluate_multi(X_test, y_test, fit, n_classes=3):
  """
  adapted from:
      https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
  """
  print(classification_report(y_test, fit.predict(X_test)))
  
  try:
    y_score = fit.decision_function(X_test)
  except:
    y_score = fit.predict_proba(X_test)
      
  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])    

  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

  # Finally average it and compute AUC
  mean_tpr /= n_classes

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  # Plot all ROC curves
  plt.figure(figsize=(12,8))
  plt.plot(fpr["micro"], tpr["micro"],
           label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["micro"]),
           color='deeppink', linestyle=':', linewidth=4)

  plt.plot(fpr["macro"], tpr["macro"],
           label='macro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["macro"]),
           color='navy', linestyle=':', linewidth=4)

  lw=2
  colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
  for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Multiclass Bins')
  plt.legend(loc="lower right")
  plt.show()

def mpNumCoEvents(closeIdx,t1,molecule):
  '''
  Compute the number of concurrent events per bar.
  +molecule[0] is the date of the first event on which the weight will be computed
  +molecule[-1] is the date of the last event on which the weight will be computed
  
  Any event that starts before t1[modelcule].max() impacts the count.
  '''
  #1) find events that span the period [molecule[0],molecule[-1]]
  t1=t1.fillna(closeIdx[-1]) # unclosed events still must impact other weights
  t1=t1[t1>=molecule[0]] # events that end at or after molecule[0]
  t1=t1.loc[:t1[molecule].max()] # events that start at or before t1[molecule].max()
  #2) count events spanning a bar
  iloc=closeIdx.searchsorted(np.array([t1.index[0],t1.max()]))
  count=pd.Series(0,index=closeIdx[iloc[0]:iloc[1]+1])
  for tIn,tOut in t1.iteritems():count.loc[tIn:tOut]+=1.
  return count.loc[molecule[0]:t1[molecule].max()]

def mpSampleTW(t1,numCoEvents,molecule):
  # Derive avg. uniqueness over the events lifespan
  wght=pd.Series(index=molecule)
  for tIn,tOut in t1.loc[wght.index].iteritems():
    wght.loc[tIn]=(1./numCoEvents.loc[tIn:tOut]).mean()
  return wght

def getIndMatrix(barIx,t1):
  # Get Indicator matrix
  indM=(pd.DataFrame(0,index=barIx,columns=range(t1.shape[0])))
  for i,(t0,t1) in enumerate(t1.iteritems()):indM.loc[t0:t1,i]=1.
  return indM

def getAvgUniqueness(indM):
  c=indM.sum(axis=1)
  u=indM.div(c, axis=0)
  avgU=u[u>0].mean()
  return avgU

def seqBootstrap(indM, sLength=None):
  if sLength is None: sLength = indM.shape[1]
  phi = []
  while len(phi) < sLength:
    avgU = pd.Series()
    for i in indM:
      indM_ = indM[phi+[i]]
      avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
    prob = avgU / avgU.sum()
    phi += [np.random.choice(indM.columns, p=prob)]
  return phi

def getAvgUniqueness(indM):
  # Average uniqueness from indicator matrix
  c=indM.sum(axis=1) # concurrency
  u=indM.div(c,axis=0) # uniqueness
  avgU=u[u>0].mean() # avg. uniqueness
  return avgU

def seqBootstrap(indM,sLength=None):
  # Generate a sample via sequential bootstrap
  if sLength is None:sLength=indM.shape[1]
  phi=[]
  while len(phi)<sLength:
    avgU=pd.Series()
    for i in indM:
      indM_=indM[phi+[i]] # reduce indM
      avgU.loc[i]=getAvgUniqueness(indM_).iloc[-1]
    prob=avgU/avgU.sum() # draw prob
    phi+=[np.random.choice(indM.columns,p=prob)]
  return phi

def mpSampleW(t1, numCoEvents, close, molecule):
  # Derive sample weight by return attribution
  ret=np.log(close).diff() # log-returns, so that they are additive
  wght=pd.Series(index=molecule)
  for tIn,tOut in t1.loc[wght.index].iteritems():
      wght.loc[tIn]=(ret.loc[tIn:tOut]/numCoEvents.loc[tIn:tOut]).sum()
  return wght.abs()

def getTimeDecay(tW,clfLastW=1.):
  # apply piecewise-linear decay to observed uniqueness (tW)
  # newest observation gets weight=1, oldest observation gets weight=clfLastW
  clfW=tW.sort_index().cumsum()
  if clfLastW>=0: slope=(1.-clfLastW)/clfW.iloc[-1]
  else: slope=1./((clfLastW+1)*clfW.iloc[-1])
  const=1.-slope*clfW.iloc[-1]
  clfW=const+slope*clfW
  clfW[clfW<0]=0
  return clfW

def getTrainTimes(t1,testTimes):
  """
  Given testTimes, find the times of the training observations
  -t1.index: Time when the observation started
  -t1.value: Time when the observation ended
  -testTimes: Times of testing observations
  """
  trn=t1.copy(deep=True)
  for i,j in testTimes.iteritems():
    df0=trn[(i<=trn.index)&(trn.index<=j)].index # train starts within test
    df1=trn[(i<=trn)&(trn<=j)].index # train ends within test
    df2=trn[(trn.index<=i)&(j<=trn)].index # train envelops test
    trn=trn.drop(df0.union(df1).union(df2))
  return trn


def getEmbargoTimes(times,pctEmbargo):
  # Get embargo time for each bar
  step=int(times.shape[0]*pctEmbargo)
  if step==0:
    mbrg=pd.Series(times,index=times)
  else:
    mbrg=pd.Series(times[step:],index=times[:-step])
    mbrg=mbrg.append(pd.Series(times[-1],index=times[-step:]))
  return mbrg

class PurgedKFold(_BaseKFold):
  """
  Extend KFold class to work with labels that span intervals
  The train is purged of observations overlapping test-label intervals
  Test set is assumed contiguous (shuffle=False), w/o training samples in between
  """
  def __init__(self,n_splits=3,t1=None,pctEmbargo=0.):
    if not isinstance(t1,pd.Series):
      raise ValueError('Label Through Dates must be a pd.Series')
    super(PurgedKFold,self).__init__(n_splits,shuffle=False,random_state=None)
    self.t1=t1
    self.pctEmbargo=pctEmbargo
      
  def split(self,X,y=None,groups=None):
    if (X.index==self.t1.index).sum()!=len(self.t1):
      raise ValueError('X and ThruDateValues must have the same index')
    indices=np.arange(X.shape[0])
    mbrg=int(X.shape[0]*self.pctEmbargo)
    test_starts=[(i[0],i[-1]+1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
    for i,j in test_starts:
      t0=self.t1.index[i] # start of test set
      test_indices=indices[i:j]
      maxT1Idx=self.t1.index.searchsorted(self.t1.index[test_indices].max())
      train_indices=self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
      if maxT1Idx<X.shape[0]: # right train (with embargo)
        train_indices=np.concatenate((train_indices, indices[maxT1Idx+mbrg:]))
      yield train_indices,test_indices
            

def cvScore(clf,X,y,sample_weight,scoring='neg_log_loss',
            t1=None,cv=None,cvGen=None,pctEmbargo=None):
  if scoring not in ['neg_log_loss','accuracy']:
      raise Exception('wrong scoring method.')
  idx = pd.IndexSlice
  if cvGen is None:
    cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
  score=[]
  for train,test in cvGen.split(X=X):
    fit=clf.fit(X=X.iloc[idx[train],:],y=y.iloc[idx[train]],
                sample_weight=sample_weight.iloc[idx[train]].values)
    if scoring=='neg_log_loss':
      prob=fit.predict_proba(X.iloc[idx[test],:])
      score_=-log_loss(y.iloc[idx[test]], prob,
                              sample_weight=sample_weight.iloc[idx[test]].values,
                              labels=clf.classes_)
    else:
      pred=fit.predict(X.iloc[idx[test],:])
      score_=accuracy_score(y.iloc[idx[test]],pred,
                            sample_weight=sample_weight.iloc[idx[test]].values)
    score.append(score_)
  return np.array(score)

def crossValPlot(skf,classifier,X_,y_):
  X = np.asarray(X_)
  y = np.asarray(y_)
  
  tprs = []
  aucs = []
  mean_fpr = np.linspace(0, 1, 100)
  
  f,ax = plt.subplots(figsize=(10,7))
  i = 0
  for train, test in skf.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    ax.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1

  ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
           label='Luck', alpha=.8)

  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  std_auc = np.std(aucs)
  ax.plot(mean_fpr, mean_tpr, color='b',
           label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
           lw=2, alpha=.8)

  std_tpr = np.std(tprs, axis=0)
  tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
  tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
  ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                   label=r'$\pm$ 1 std. dev.')

  ax.set_xlim([-0.05, 1.05])
  ax.set_ylim([-0.05, 1.05])
  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('True Positive Rate')
  ax.set_title('Receiver operating characteristic example')
  ax.legend(bbox_to_anchor=(1,1))
    
def crossValPlot2(skf,classifier,X,y):
  tprs = []
  aucs = []
  mean_fpr = np.linspace(0, 1, 100)
  idx = pd.IndexSlice
  f,ax = plt.subplots(figsize=(10,7))
  i = 0
  for train, test in skf.split(X, y):
    probas_ = (classifier.fit(X.iloc[idx[train]], y.iloc[idx[train]])
               .predict_proba(X.iloc[idx[test]]))
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y.iloc[idx[test]], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    ax.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1

  ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
           label='Luck', alpha=.8)

  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  std_auc = np.std(aucs)
  ax.plot(mean_fpr, mean_tpr, color='b',
           label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
           lw=2, alpha=.8)

  std_tpr = np.std(tprs, axis=0)
  tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
  tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
  ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                   label=r'$\pm$ 1 std. dev.')

  ax.set_xlim([-0.05, 1.05])
  ax.set_ylim([-0.05, 1.05])
  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('True Positive Rate')
  ax.set_title('Receiver operating characteristic example')
  ax.legend(bbox_to_anchor=(1,1))

def featImpMDI(fit, featNames):
  df0={i:tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
  df0=pd.DataFrame.from_dict(df0, orient='index')
  df0.columns=featNames
  df0=df0.replace(0, np.nan) # because max_features = 1
  imp=pd.concat({'mean':df0.mean(), 'std':df0.std()*df0.shape[0]**-0.5}, axis=1)
  imp/=imp['mean'].sum()
  return imp

def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring='neg_log_loss'):
  # Feat importance based on OOS score reduction
  if scoring not in ['neg_log_loss', 'accuracy']:
      raise Exception('Wrong scoring method.')

  cvGen=PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo) # purged cv
  scr0, scr1= pd.Series(), pd.DataFrame(columns=X.columns)
  for i, (train, test) in enumerate(cvGen.split(X=X)):
    X0,y0,w0= X.iloc[train,:], y.iloc[train], sample_weight.iloc[train]
    X1,y1,w1= X.iloc[test,:], y.iloc[test], sample_weight.iloc[test]
    fit=clf.fit(X=X0, y=y0, sample_weight=w0.values)
    if scoring=='neg_log_loss':
      prob=fit.predict_proba(X1)
      scr0.loc[i]=-log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
    else:
      pred=fit.predict(X1)
      scr0.loc[i]=accuracy_score(y1, pred, sample_weight=w1.values)
    for j in X.columns:
      X1_=X1.copy(deep=True)
      np.random.shuffle(X1_[j].values) # Permutation of a single column
      if scoring=='neg_log_loss':
        prob=fit.predict_proba(X1_)
        src1.loc[i,j]=-log_loss(y1,prob,sample_weight=w1.values,
                                labels=clf.classes_)
      else:
        pred=fit.predict(X1_)
        scr1.loc[i,j]=accuracy_score(y1,pred,sample_weight=w1.values)
  imp=(-scr1).add(scr0, axis=0)
  if scoring=='neg_log_loss': imp=imp/-scr1
  else: imp=imp/(1.-scr1)
  imp=pd.concat({'mean':imp.mean(), 'std':imp.std()*imp.shape[0]**-0.5}, axis=1)
  return imp, scr0.mean()

def featImportance(trnsX, cont, clf, fit, cv=10, pctEmbargo=0, scoring='accuracy', method='MDI'):
  oob = fit.oob_score_
  if method=='MDI':
    imp=featImpMDI(fit, featNames=trnsX.columns)
    oos=cvScore(clf, X=trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'], t1=cont['t1'], pctEmbargo=pctEmbargo, scoring=scoring).mean()
  elif method=='MDA':
    imp, oos = featImpMDA(clf, X=trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'], t1=cont['t1'], pctEmbargo=pctEmbargo, scoring=scoring)
  else:
    raise Exception('method is invalid.')
  return imp, oob, oos

def plot_feature_importance(importance_df, oob_score, oos_score, save_fig=False, output_path=None):
  """
  Snippet 8.10, page 124. Feature importance plotting function.

  Plot feature importance.

  :param importance_df: (pd.DataFrame): Mean and standard deviation feature importance.
  :param oob_score: (float): Out-of-bag score.
  :param oos_score: (float): Out-of-sample (or cross-validation) score.
  :param save_fig: (bool): Boolean flag to save figure to a file.
  :param output_path: (str): If save_fig is True, path where figure should be saved.
  """
  # Plot mean imp bars with std
  plt.figure(figsize=(10, importance_df.shape[0] / 5))
  importance_df.sort_values('mean', ascending=True, inplace=True)
  importance_df['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=importance_df['std'], error_kw={'ecolor': 'r'})
  plt.title('Feature importance. OOB Score:{}; OOS score:{}'.format(round(oob_score, 4), round(oos_score, 4)))

  if save_fig is True:
    plt.savefig(output_path)
  else:
    plt.show()

def relative_strength_index(df, n, high_col, low_col):
  """Calculate Relative Strength Index(RSI) for given data.
  https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py
  https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises/blob/master/notebooks/mlfinlab/corefns/financial_functions.py#L22-L53

  :param df: pandas.DataFrame
  :param n: 
  :return: pandas.DataFrame
  """
  i = 0
  UpI = [0]
  DoI = [0]
  while i + 1 <= df.index[-1]:
    UpMove = df.loc[i + 1, high_col] - df.loc[i, high_col]
    DoMove = df.loc[i, low_col] - df.loc[i + 1, low_col]
    if UpMove > DoMove and UpMove > 0:
      UpD = UpMove
    else:
      UpD = 0
    UpI.append(UpD)
    if DoMove > UpMove and DoMove > 0:
      DoD = DoMove
    else:
      DoD = 0
    DoI.append(DoD)
    i = i + 1
  UpI = pd.Series(UpI)
  DoI = pd.Series(DoI)
  PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
  NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
  RSI = pd.Series(round(PosDI * 100. / (PosDI + NegDI)), name='RSI_' + str(n))
  # df = df.join(RSI)
  return RSI

def get_rsi(df, window=14, high_col='High', low_col='Low'):
  df = df.copy(deep=True).reset_index()
  rsi = relative_strength_index(df, window, high_col, low_col)
  rsi_df = pd.Series(data=rsi.values, index=df.index)
  return rsi_df

def add_rsi_to(df, windows):
  for w in windows:
    rsi = get_rsi(df, window=w).squeeze()
    df[f'rsi_{w}'] = rsi
        
def add_momentum_to(df, windows):
  for w in windows:
    df[f'mom_{w}'] = df['Close'].pct_change(w)

def add_volatility_to(df, windows):
  for w in windows:
    df[f'vol_{w}'] = (df['log_ret'].rolling(window=w, min_periods=w, center=False).std())

def add_autocorrelation_to(df, autocorrelation_window, autocorrelation_orders):
  for o in autocorrelation_orders:
    df[f'autocorr_{o}'] = (df['log_ret'].rolling(window=autocorrelation_window,
                                                 min_periods=autocorrelation_window,
                                                 center=False).
                           apply(lambda x: x.autocorr(lag=o), raw=False))

class EvaluateStrategy:
  '''
  This class allows to evaluate a strategy once the model is finished.
  It will allow you to:
  - Step the model to trade one at a time.
  - Run it bulk mode, i.e. all at once.
  - Arm or disarm your entire positions.
  
  It runs based on the following premises:
  - You have a standard and fixed cash reference. All bet sizes are referred to it.
  - Trade costs are not included (not needed at the moment) but will definitely play an important role.
  
  At the end, you can get some stats as a dictionary that tells a bit about the progression of your model.
  '''
  def __init__(self, cash_ref, cash0, security0, price0):
    self.cash_ref = cash_ref
    self.cash = cash0
    self.security = security0
    self.price = price0
    self.stats = {'initial_valuation': self.get_current_value(), 'r': 0., 'net_profit':0., 
                  'cash_ref': cash_ref, 'cash': cash0, 'security': security0, 'price': price0, 
                  'valuation': self.get_current_value(), 'n_trades': 0}
  
  def _update_stats(self):
    curr_valuation = self.get_current_value()
    self.stats['r'] = (curr_valuation - self.stats['initial_valuation']) / self.stats['initial_valuation']
    self.stats['net_profit'] = curr_valuation - self.stats['initial_valuation']
    self.stats['cash'] = self.cash
    self.stats['security'] = self.security
    self.stats['price'] = self.price
    self.stats['valuation'] = curr_valuation
    self.stats['n_trades'] += 1
      
  def get_status(self):
    return self.cash, self.security, self.price
  
  def get_current_value(self):
    return self.cash + self.security * self.price
  
  def get_stats(self):
    return self.stats
  
  def trade(self, bet, size, price):
    current_security_value = self.security * price
    traded_money = min(current_security_value if bet == -1 else self.cash, size * self.cash_ref)
    self.security = (current_security_value + float(bet) * traded_money) / price
    self.cash = self.cash + float(-bet) * traded_money
    self.price = price
    self._update_stats()
  
  def buy_all(self, price):
    current_security_value = self.security * price
    self.security = (current_security_value + self.cash) / price
    self.cash = 0.
    self.price = price
    self._update_stats()
      
  def sell_all(self, price):
    self.cash += self.security * price
    self.security = 0.
    self.price = price
    self._update_stats()
      
  def bulk_trade(self, environment_df, enable_log=False):
    for index, row in environment_df.iterrows():
      self.trade(row['side'], row['bet_size'], row['price'])
      if enable_log: display(self.get_status(), self.get_current_value())
