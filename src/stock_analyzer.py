# -*- coding: utf-8 -*-
"""
趋势交易分析器 - 基于用户交易理念
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum

import pandas as pd
import numpy as np

from src.config import get_config

logger = logging.getLogger(__name__)


class TrendStatus(Enum):
    STRONG_BULL = "强势多头"      # MA5 > MA10 > MA20，且间距扩大
    BULL = "多头排列"             # MA5 > MA10 > MA20
    WEAK_BULL = "弱势多头"        # MA5 > MA10，但 MA10 < MA20
    CONSOLIDATION = "盘整"        # 均线缠绕
    WEAK_BEAR = "弱势空头"        # MA5 < MA10，但 MA10 > MA20
    BEAR = "空头排列"             # MA5 < MA10 < MA20
    STRONG_BEAR = "强势空头"      # MA5 < MA10 < MA20，且间距扩大


class VolumeStatus(Enum):
    HEAVY_VOLUME_UP = "放量上涨"       # 量价齐升
    HEAVY_VOLUME_DOWN = "放量下跌"     # 放量杀跌
    SHRINK_VOLUME_UP = "缩量上涨"      # 无量上涨
    SHRINK_VOLUME_DOWN = "缩量回调"    # 缩量回调（好）
    NORMAL = "量能正常"


class BuySignal(Enum):
    STRONG_BUY = "强烈买入"       # 多条件满足
    BUY = "买入"                  # 基本条件满足
    HOLD = "持有"                 # 已持有可继续
    WAIT = "观望"                 # 等待更好时机
    SELL = "卖出"                 # 趋势转弱
    STRONG_SELL = "强烈卖出"      # 趋势破坏


class MACDStatus(Enum):
    GOLDEN_CROSS_ZERO = "零轴上金叉"      # DIF上穿DEA，且在零轴上方
    GOLDEN_CROSS = "金叉"                # DIF上穿DEA
    BULLISH = "多头"                    # DIF>DEA>0
    CROSSING_UP = "上穿零轴"             # DIF上穿零轴
    CROSSING_DOWN = "下穿零轴"           # DIF下穿零轴
    BEARISH = "空头"                    # DIF<DEA<0
    DEATH_CROSS = "死叉"                # DIF下穿DEA


class RSIStatus(Enum):
    OVERBOUGHT = "超买"        # RSI > 70
    STRONG_BUY = "强势买入"    # 50 < RSI < 70
    NEUTRAL = "中性"          # 40 <= RSI <= 60
    WEAK = "弱势"             # 30 < RSI < 40
    OVERSOLD = "超卖"         # RSI < 30


@dataclass
class TrendAnalysisResult:
    """趋势分析结果"""
    code: str
    
    trend_status: TrendStatus = TrendStatus.CONSOLIDATION
    ma_alignment: str = ""
    trend_strength: float = 0.0
    
    ma5: float = 0.0
    ma10: float = 0.0
    ma20: float = 0.0
    ma60: float = 0.0
    current_price: float = 0.0
    
    bias_ma5: float = 0.0
    bias_ma10: float = 0.0
    bias_ma20: float = 0.0
    
    volume_status: VolumeStatus = VolumeStatus.NORMAL
    volume_ratio_5d: float = 0.0
    volume_trend: str = ""
    
    support_ma5: bool = False
    support_ma10: bool = False
    resistance_levels: List[float] = field(default_factory=list)
    support_levels: List[float] = field(default_factory=list)
    
    macd_dif: float = 0.0
    macd_dea: float = 0.0
    macd_bar: float = 0.0
    macd_status: MACDStatus = MACDStatus.BULLISH
    macd_signal: str = ""
    
    rsi_6: float = 0.0
    rsi_12: float = 0.0
    rsi_24: float = 0.0
    rsi_status: RSIStatus = RSIStatus.NEUTRAL
    rsi_signal: str = ""
    
    buy_signal: BuySignal = BuySignal.WAIT
    signal_score: int = 0
    signal_reasons: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)

    # 附加形态信号（仅 US、价格>=50 时触发）
    current_usd_price: float = 0.0
    unusual_volume: bool = False
    bearish_reversal: bool = False
    bullish_divergence: bool = False
    pattern_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'trend_status': self.trend_status.value,
            'ma_alignment': self.ma_alignment,
            'trend_strength': self.trend_strength,
            'ma5': self.ma5,
            'ma10': self.ma10,
            'ma20': self.ma20,
            'ma60': self.ma60,
            'current_price': self.current_price,
            'bias_ma5': self.bias_ma5,
            'bias_ma10': self.bias_ma10,
            'bias_ma20': self.bias_ma20,
            'volume_status': self.volume_status.value,
            'volume_ratio_5d': self.volume_ratio_5d,
            'volume_trend': self.volume_trend,
            'support_ma5': self.support_ma5,
            'support_ma10': self.support_ma10,
            'buy_signal': self.buy_signal.value,
            'signal_score': self.signal_score,
            'signal_reasons': self.signal_reasons,
            'risk_factors': self.risk_factors,
            'macd_dif': self.macd_dif,
            'macd_dea': self.macd_dea,
            'macd_bar': self.macd_bar,
            'macd_status': self.macd_status.value,
            'macd_signal': self.macd_signal,
            'rsi_6': self.rsi_6,
            'rsi_12': self.rsi_12,
            'rsi_24': self.rsi_24,
            'rsi_status': self.rsi_status.value,
            'rsi_signal': self.rsi_signal,
            'current_usd_price': self.current_usd_price,
            'unusual_volume': bool(self.unusual_volume),
            'bearish_reversal': bool(self.bearish_reversal),
            'bullish_divergence': bool(self.bullish_divergence),
            'pattern_flags': list(self.pattern_flags),
        }


class StockTrendAnalyzer:
    """
    股票趋势分析器

    基于用户交易理念实现：
    1. 趋势判断 - MA5>MA10>MA20 多头排列
    2. 乖离率检测 - 不追高，偏离 MA5 超过 5% 不买
    3. 量能分析 - 偏好缩量回调
    4. 买点识别 - 回踩 MA5/MA10 支撑
    5. MACD 指标 - 趋势确认和金叉死叉信号
    6. RSI 指标 - 超买超卖判断
    """
    
    VOLUME_SHRINK_RATIO = 0.7
    VOLUME_HEAVY_RATIO = 1.5
    MA_SUPPORT_TOLERANCE = 0.02
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    RSI_SHORT = 6
    RSI_MID = 12
    RSI_LONG = 24
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    def __init__(self):
        pass
    
    def analyze(self, df: pd.DataFrame, code: str) -> TrendAnalysisResult:
        result = TrendAnalysisResult(code=code)
        
        if df is None or df.empty or len(df) < 20:
            logger.warning(f"{code} 数据不足，无法进行趋势分析")
            result.risk_factors.append("数据不足，无法完成分析")
            return result
        
        df = df.sort_values('date').reset_index(drop=True)
        df = self._calculate_mas(df)
        df = self._calculate_macd(df)
        df = self._calculate_rsi(df)
        
        latest = df.iloc[-1]
        result.current_price = float(latest['close'])
        result.ma5 = float(latest['MA5'])
        result.ma10 = float(latest['MA10'])
        result.ma20 = float(latest['MA20'])
        result.ma60 = float(latest.get('MA60', 0))
        try:
            result.current_usd_price = float(result.current_price)
        except (TypeError, ValueError):
            result.current_usd_price = 0.0
        
        self._analyze_trend(df, result)
        self._calculate_bias(result)
        self._analyze_volume(df, result)
        self._analyze_support_resistance(df, result)
        self._analyze_macd(df, result)
        self._analyze_rsi(df, result)
        self._generate_signal(result)
        
        return result
    
    def _calculate_mas(self, df):
        df = df.copy()
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA60'] = df['close'].rolling(window=60).mean() if len(df) >= 60 else df['MA20']
        return df

    def _calculate_macd(self, df):
        df = df.copy()
        ema_fast = df['close'].ewm(span=self.MACD_FAST, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.MACD_SLOW, adjust=False).mean()
        df['MACD_DIF'] = ema_fast - ema_slow
        df['MACD_DEA'] = df['MACD_DIF'].ewm(span=self.MACD_SIGNAL, adjust=False).mean()
        df['MACD_BAR'] = (df['MACD_DIF'] - df['MACD_DEA']) * 2
        return df

    def _calculate_rsi(self, df):
        df = df.copy()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        for period in [self.RSI_SHORT, self.RSI_MID, self.RSI_LONG]:
            avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            df[f'RSI_{period}'] = rsi.fillna(50)
        return df
    
    def _analyze_trend(self, df, result):
        ma5, ma10, ma20 = result.ma5, result.ma10, result.ma20
        
        if ma5 > ma10 > ma20:
            prev = df.iloc[-5] if len(df) >= 5 else df.iloc[-1]
            prev_spread = (prev['MA5'] - prev['MA20']) / prev['MA20'] * 100 if prev['MA20'] > 0 else 0
            curr_spread = (ma5 - ma20) / ma20 * 100 if ma20 > 0 else 0
            
            if curr_spread > prev_spread and curr_spread > 5:
                result.trend_status = TrendStatus.STRONG_BULL
                result.ma_alignment = "强势多头排列，均线发散上行"
                result.trend_strength = 90
            else:
                result.trend_status = TrendStatus.BULL
                result.ma_alignment = "多头排列 MA5>MA10>MA20"
                result.trend_strength = 75
                
        elif ma5 > ma10 and ma10 <= ma20:
            result.trend_status = TrendStatus.WEAK_BULL
            result.ma_alignment = "弱势多头，MA5>MA10 但 MA10<=MA20"
            result.trend_strength = 55
            
        elif ma5 < ma10 < ma20:
            prev = df.iloc[-5] if len(df) >= 5 else df.iloc[-1]
            prev_spread = (prev['MA20'] - prev['MA5']) / prev['MA5'] * 100 if prev['MA5'] > 0 else 0
            curr_spread = (ma20 - ma5) / ma5 * 100 if ma5 > 0 else 0
            
            if curr_spread > prev_spread and curr_spread > 5:
                result.trend_status = TrendStatus.STRONG_BEAR
                result.ma_alignment = "强势空头排列，均线发散下行"
                result.trend_strength = 10
            else:
                result.trend_status = TrendStatus.BEAR
                result.ma_alignment = "空头排列 MA5<MA10<MA20"
                result.trend_strength = 25
                
        elif ma5 < ma10 and ma10 >= ma20:
            result.trend_status = TrendStatus.WEAK_BEAR
            result.ma_alignment = "弱势空头，MA5<MA10 但 MA10>=MA20"
            result.trend_strength = 40
            
        else:
            result.trend_status = TrendStatus.CONSOLIDATION
            result.ma_alignment = "均线缠绕，趋势不明"
            result.trend_strength = 50
    
    def _calculate_bias(self, result):
        price = result.current_price
        
        if result.ma5 > 0:
            result.bias_ma5 = (price - result.ma5) / result.ma5 * 100
        if result.ma10 > 0:
            result.bias_ma10 = (price - result.ma10) / result.ma10 * 100
        if result.ma20 > 0:
            result.bias_ma20 = (price - result.ma20) / result.ma20 * 100
    
    def _analyze_volume(self, df, result):
        if len(df) < 5:
            return
        
        latest = df.iloc[-1]
        vol_5d_avg = df['volume'].iloc[-6:-1].mean()
        
        if vol_5d_avg > 0:
            result.volume_ratio_5d = float(latest['volume']) / vol_5d_avg
        
        prev_close = df.iloc[-2]['close']
        price_change = (latest['close'] - prev_close) / prev_close * 100
        
        if result.volume_ratio_5d >= self.VOLUME_HEAVY_RATIO:
            if price_change > 0:
                result.volume_status = VolumeStatus.HEAVY_VOLUME_UP
                result.volume_trend = "放量上涨，多头力量强劲"
            else:
                result.volume_status = VolumeStatus.HEAVY_VOLUME_DOWN
                result.volume_trend = "放量下跌，注意风险"
        elif result.volume_ratio_5d <= self.VOLUME_SHRINK_RATIO:
            if price_change > 0:
                result.volume_status = VolumeStatus.SHRINK_VOLUME_UP
                result.volume_trend = "缩量上涨，上攻动能不足"
            else:
                result.volume_status = VolumeStatus.SHRINK_VOLUME_DOWN
                result.volume_trend = "缩量回调，洗盘特征明显（好）"
        else:
            result.volume_status = VolumeStatus.NORMAL
            result.volume_trend = "量能正常"
    
    def _analyze_support_resistance(self, df, result):
        price = result.current_price
        
        if result.ma5 > 0:
            ma5_distance = abs(price - result.ma5) / result.ma5
            if ma5_distance <= self.MA_SUPPORT_TOLERANCE and price >= result.ma5:
                result.support_ma5 = True
                result.support_levels.append(result.ma5)
        
        if result.ma10 > 0:
            ma10_distance = abs(price - result.ma10) / result.ma10
            if ma10_distance <= self.MA_SUPPORT_TOLERANCE and price >= result.ma10:
                result.support_ma10 = True
                if result.ma10 not in result.support_levels:
                    result.support_levels.append(result.ma10)
        
        if result.ma20 > 0 and price >= result.ma20:
            result.support_levels.append(result.ma20)
        
        if len(df) >= 20:
            recent_high = df['high'].iloc[-20:].max()
            if recent_high > price:
                result.resistance_levels.append(recent_high)

    def _analyze_macd(self, df, result):
        if len(df) < self.MACD_SLOW:
            result.macd_signal = "数据不足"
            return

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        result.macd_dif = float(latest['MACD_DIF'])
        result.macd_dea = float(latest['MACD_DEA'])
        result.macd_bar = float(latest['MACD_BAR'])

        prev_dif_dea = prev['MACD_DIF'] - prev['MACD_DEA']
        curr_dif_dea = result.macd_dif - result.macd_dea

        is_golden_cross = prev_dif_dea <= 0 and curr_dif_dea > 0
        is_death_cross = prev_dif_dea >= 0 and curr_dif_dea < 0

        is_crossing_up = prev['MACD_DIF'] <= 0 and result.macd_dif > 0
        is_crossing_down = prev['MACD_DIF'] >= 0 and result.macd_dif < 0

        if is_golden_cross and result.macd_dif > 0:
            result.macd_status = MACDStatus.GOLDEN_CROSS_ZERO
            result.macd_signal = "⭐ 零轴上金叉，强烈买入信号！"
        elif is_crossing_up:
            result.macd_status = MACDStatus.CROSSING_UP
            result.macd_signal = "⚡ DIF上穿零轴，趋势转强"
        elif is_golden_cross:
            result.macd_status = MACDStatus.GOLDEN_CROSS
            result.macd_signal = "✅ 金叉，趋势向上"
        elif is_death_cross:
            result.macd_status = MACDStatus.DEATH_CROSS
            result.macd_signal = "❌ 死叉，趋势向下"
        elif is_crossing_down:
            result.macd_status = MACDStatus.CROSSING_DOWN
            result.macd_signal = "⚠️ DIF下穿零轴，趋势转弱"
        elif result.macd_dif > 0 and result.macd_dea > 0:
            result.macd_status = MACDStatus.BULLISH
            result.macd_signal = "✓ 多头排列，持续上涨"
        elif result.macd_dif < 0 and result.macd_dea < 0:
            result.macd_status = MACDStatus.BEARISH
            result.macd_signal = "⚠ 空头排列，持续下跌"
        else:
            result.macd_status = MACDStatus.BULLISH
            result.macd_signal = " MACD 中性区域"

    def _analyze_rsi(self, df, result):
        if len(df) < self.RSI_LONG:
            result.rsi_signal = "数据不足"
            return

        latest = df.iloc[-1]
        result.rsi_6 = float(latest[f'RSI_{self.RSI_SHORT}'])
        result.rsi_12 = float(latest[f'RSI_{self.RSI_MID}'])
        result.rsi_24 = float(latest[f'RSI_{self.RSI_LONG}'])

        rsi_mid = result.rsi_12

        if rsi_mid > self.RSI_OVERBOUGHT:
            result.rsi_status = RSIStatus.OVERBOUGHT
            result.rsi_signal = f"⚠️ RSI超买({rsi_mid:.1f}>70)，短期回调风险高"
        elif rsi_mid > 60:
            result.rsi_status = RSIStatus.STRONG_BUY
            result.rsi_signal = f"✅ RSI强势({rsi_mid:.1f})，多头力量充足"
        elif rsi_mid >= 40:
            result.rsi_status = RSIStatus.NEUTRAL
            result.rsi_signal = f" RSI中性({rsi_mid:.1f})，震荡整理中"
        elif rsi_mid >= self.RSI_OVERSOLD:
            result.rsi_status = RSIStatus.WEAK
            result.rsi_signal = f"⚡ RSI弱势({rsi_mid:.1f})，关注反弹"
        else:
            result.rsi_status = RSIStatus.OVERSOLD
            result.rsi_signal = f"⭐ RSI超卖({rsi_mid:.1f}<30)，反弹机会大"

    def _generate_signal(self, result):
        score = 0
        reasons = []
        risks = []

        trend_scores = {
            TrendStatus.STRONG_BULL: 30,
            TrendStatus.BULL: 26,
            TrendStatus.WEAK_BULL: 18,
            TrendStatus.CONSOLIDATION: 12,
            TrendStatus.WEAK_BEAR: 8,
            TrendStatus.BEAR: 4,
            TrendStatus.STRONG_BEAR: 0,
        }
        trend_score = trend_scores.get(result.trend_status, 12)
        score += trend_score

        if result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL]:
            reasons.append(f"✅ {result.trend_status.value}，顺势做多")
        elif result.trend_status in [TrendStatus.BEAR, TrendStatus.STRONG_BEAR]:
            risks.append(f"⚠️ {result.trend_status.value}，不宜做多")

        bias = result.bias_ma5
        if bias != bias or bias is None:
            bias = 0.0
        base_threshold = get_config().bias_threshold
        trend_strength = result.trend_strength if result.trend_strength == result.trend_strength else 0.0
        if result.trend_status == TrendStatus.STRONG_BULL and (trend_strength or 0) >= 70:
            effective_threshold = base_threshold * 1.5
            is_strong_trend = True
        else:
            effective_threshold = base_threshold
            is_strong_trend = False

        if bias < 0:
            if bias > -3:
                score += 20
                reasons.append(f"✅ 价格略低于MA5({bias:.1f}%)，回踩买点")
            elif bias > -5:
                score += 16
                reasons.append(f"✅ 价格回踩MA5({bias:.1f}%)，观察支撑")
            else:
                score += 8
                risks.append(f"⚠️ 乖离率过大({bias:.1f}%)，可能破位")
        elif bias < 2:
            score += 18
            reasons.append(f"✅ 价格贴近MA5({bias:.1f}%)，介入好时机")
        elif bias < base_threshold:
            score += 14
            reasons.append(f"⚡ 价格略高于MA5({bias:.1f}%)，可小仓介入")
        elif bias > effective_threshold:
            score += 4
            risks.append(
                f"❌ 乖离率过高({bias:.1f}%>{effective_threshold:.1f}%)，严禁追高！"
            )
        elif bias > base_threshold and is_strong_trend:
            score += 10
            reasons.append(
                f"⚡ 强势趋势中乖离率偏高({bias:.1f}%)，可轻仓追踪"
            )
        else:
            score += 4
            risks.append(
                f"❌ 乖离率过高({bias:.1f}%>{base_threshold:.1f}%)，严禁追高！"
            )

        volume_scores = {
            VolumeStatus.SHRINK_VOLUME_DOWN: 15,
            VolumeStatus.HEAVY_VOLUME_UP: 12,
            VolumeStatus.NORMAL: 10,
            VolumeStatus.SHRINK_VOLUME_UP: 6,
            VolumeStatus.HEAVY_VOLUME_DOWN: 0,
        }
        vol_score = volume_scores.get(result.volume_status, 8)
        score += vol_score

        if result.volume_status == VolumeStatus.SHRINK_VOLUME_DOWN:
            reasons.append("✅ 缩量回调，主力洗盘")
        elif result.volume_status == VolumeStatus.HEAVY_VOLUME_DOWN:
            risks.append("⚠️ 放量下跌，注意风险")

        if result.support_ma5:
            score += 5
            reasons.append("✅ MA5支撑有效")
        if result.support_ma10:
            score += 5
            reasons.append("✅ MA10支撑有效")

        macd_scores = {
            MACDStatus.GOLDEN_CROSS_ZERO: 15,
            MACDStatus.GOLDEN_CROSS: 12,
            MACDStatus.CROSSING_UP: 10,
            MACDStatus.BULLISH: 8,
            MACDStatus.BEARISH: 2,
            MACDStatus.CROSSING_DOWN: 0,
            MACDStatus.DEATH_CROSS: 0,
        }
        score += macd_scores.get(result.macd_status, 5)

        if result.macd_status in [MACDStatus.GOLDEN_CROSS_ZERO, MACDStatus.GOLDEN_CROSS]:
            reasons.append(f"✅ {result.macd_signal}")
        elif result.macd_status in [MACDStatus.DEATH_CROSS, MACDStatus.CROSSING_DOWN]:
            risks.append(f"⚠️ {result.macd_signal}")
        else:
            reasons.append(result.macd_signal)

        rsi_scores = {
            RSIStatus.OVERSOLD: 10,
            RSIStatus.STRONG_BUY: 8,
            RSIStatus.NEUTRAL: 5,
            RSIStatus.WEAK: 3,
            RSIStatus.OVERBOUGHT: 0,
        }
        score += rsi_scores.get(result.rsi_status, 5)

        if result.rsi_status in [RSIStatus.OVERSOLD, RSIStatus.STRONG_BUY]:
            reasons.append(f"✅ {result.rsi_signal}")
        elif result.rsi_status == RSIStatus.OVERBOUGHT:
            risks.append(f"⚠️ {result.rsi_signal}")
        else:
            reasons.append(result.rsi_signal)

        market_flags = getattr(result, 'market_pattern_flags', None) or []
        for text in market_flags:
            text = str(text or '').strip()
            if not text:
                continue
            if any(key in text for key in ['反转', '死叉', '跌破']):
                risks.append(f"⚠️ {text}")
                score = max(0, score - 8)
            elif '背离' in text or '放量下跌' in text:
                risks.append(f"⚠️ {text}")
                score = max(0, score - 6)

        result.signal_score = score
        result.signal_reasons = reasons
        result.risk_factors = risks

        if score >= 75 and result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL]:
            result.buy_signal = BuySignal.STRONG_BUY
        elif score >= 60 and result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL, TrendStatus.WEAK_BULL]:
            result.buy_signal = BuySignal.BUY
        elif score >= 45:
            result.buy_signal = BuySignal.HOLD
        elif score >= 30:
            result.buy_signal = BuySignal.WAIT
        elif result.trend_status in [TrendStatus.BEAR, TrendStatus.STRONG_BEAR]:
            result.buy_signal = BuySignal.STRONG_SELL
        else:
            result.buy_signal = BuySignal.SELL
    
    def format_analysis(self, result):
        lines = [
            f"=== {result.code} 趋势分析 ===",
            f"",
            f"📊 趋势判断: {result.trend_status.value}",
            f"   均线排列: {result.ma_alignment}",
            f"   趋势强度: {result.trend_strength}/100",
            f"",
            f"📈 均线数据:",
            f"   现价: {result.current_price:.2f}",
            f"   MA5:  {result.ma5:.2f} (乖离 {result.bias_ma5:+.2f}%)",
            f"   MA10: {result.ma10:.2f} (乖离 {result.bias_ma10:+.2f}%)",
            f"   MA20: {result.ma20:.2f} (乖离 {result.bias_ma20:+.2f}%)",
            f"",
            f"📊 量能分析: {result.volume_status.value}",
            f"   量比(vs5日): {result.volume_ratio_5d:.2f}",
            f"   量能趋势: {result.volume_trend}",
            f"",
            f"📈 MACD指标: {result.macd_status.value}",
            f"   DIF: {result.macd_dif:.4f}",
            f"   DEA: {result.macd_dea:.4f}",
            f"   MACD: {result.macd_bar:.4f}",
            f"   信号: {result.macd_signal}",
            f"",
            f"📊 RSI指标: {result.rsi_status.value}",
            f"   RSI(6): {result.rsi_6:.1f}",
            f"   RSI(12): {result.rsi_12:.1f}",
            f"   RSI(24): {result.rsi_24:.1f}",
            f"   信号: {result.rsi_signal}",
            f"",
            f"🎯 操作建议: {result.buy_signal.value}",
            f"   综合评分: {result.signal_score}/100",
        ]

        if result.signal_reasons:
            lines.append(f"")
            lines.append(f"✅ 买入理由:")
            for reason in result.signal_reasons:
                lines.append(f"   {reason}")

        if result.risk_factors:
            lines.append(f"")
            lines.append(f"⚠️ 风险因素:")
            for risk in result.risk_factors:
                lines.append(f"   {risk}")

        return "\n".join(lines)


def analyze_stock(df: pd.DataFrame, code: str) -> TrendAnalysisResult:
    analyzer = StockTrendAnalyzer()
    return analyzer.analyze(df, code)


def _is_us_code(code: str) -> bool:
    c = str(code or "").strip().upper()
    return c.endswith(".US") or c.startswith("US_") or (len(c) >= 2 and c[:2] == "US")


class MarketPatternDetector:
    UNUSUAL_VOLUME_MIN_RATIO = 2.0
    DIVERGENCE_LOOKBACK = 8
    REVERSAL_LOOKBACK = 8

    def run(self, df: pd.DataFrame, result: "TrendAnalysisResult") -> None:
        if not _is_us_code(result.code):
            return
        if result.current_usd_price < 50:
            return
        if df is None or df.empty or len(df) < self.REVERSAL_LOOKBACK + 1:
            return
        flags = []
        if self._check_unusual_volume(df):
            result.unusual_volume = True
            flags.append("异常放量")
        if self._check_bearish_reversal(df):
            result.bearish_reversal = True
            flags.append("看跌反转")
        if self._check_macd_divergence(df):
            result.bullish_divergence = True
            flags.append("MACD背离")
        if flags:
            result.pattern_flags.extend(flags)

    def _check_unusual_volume(self, df: pd.DataFrame) -> bool:
        if len(df) < 6 or "volume" not in df.columns:
            return False
        last = float(df.iloc[-1]["volume"])
        avg_prev = float(df.iloc[-6:-1]["volume"].mean())
        ratio = last / avg_prev if avg_prev > 0 else 0.0
        return ratio >= self.UNUSUAL_VOLUME_MIN_RATIO

    def _check_bearish_reversal(self, df: pd.DataFrame) -> bool:
        if len(df) < self.REVERSAL_LOOKBACK + 1:
            return False
        if not all(col in df.columns for col in ["open", "high", "low", "close"]):
            return False
        today_open = float(df.iloc[-1]["open"])
        today_close = float(df.iloc[-1]["close"])
        high = float(df.iloc[-1]["high"])
        low = float(df.iloc[-1]["low"])
        body_range = high - low
        if body_range <= 0:
            return False
        return (
            today_close < today_open
            and abs(today_close - today_open) / body_range >= 0.55
            and (high - max(today_open, today_close)) / body_range >= 0.18
        )

    def _check_macd_divergence(self, df: pd.DataFrame) -> bool:
        if len(df) < self.DIVERGENCE_LOOKBACK:
            return False
        if not all(col in df.columns for col in ["MACD_DIF", "MACD_DEA", "close"]):
            return False
        block = df.tail(self.DIVERGENCE_LOOKBACK)
        hist = (block["MACD_DIF"].astype(float) - block["MACD_DEA"].astype(float))
        close = block["close"].astype(float)
        n = len(block)
        if n < 4:
            return False
        mid = n // 2
        prev_hist_min = hist.iloc[:mid].min()
        curr_hist_min = hist.iloc[mid:].min()
        prev_close_min = close.iloc[:mid].min()
        curr_close_min = close.iloc[mid:].min()
        return bool(curr_hist_min < prev_hist_min and curr_close_min > prev_close_min)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import numpy as np

    dates = pd.date_range(start="2025-01-01", periods=60, freq="D")
    np.random.seed(42)

    base_price = 50.0
    prices = [base_price]
    for _ in range(59):
        prices.append(prices[-1] * (1 + np.random.randn() * 0.01))

    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "close": prices,
            "volume": [np.random.randint(1_000_000, 5_000_000) for _ in prices],
        }
    )
    df["MACD_DIF"] = np.nan
    df["MACD_DEA"] = np.nan

    result = StockTrendAnalyzer().analyze(df, "US.AAPL")
    print(StockTrendAnalyzer().format_analysis(result))
    detector = MarketPatternDetector()
    detector.run(df, result)
    print({"pattern_flags": result.pattern_flags, "unusual_volume": result.unusual_volume, "bearish_reversal": result.bearish_reversal, "bullish_divergence": result.bullish_divergence})
