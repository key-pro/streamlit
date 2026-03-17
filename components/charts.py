import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_candlestick_chart(df: pd.DataFrame, ticker_symbol: str, overlays: list = None, oscillators: list = None) -> go.Figure:
    """
    TradingView風の高度なインタラクティブなPlotlyローソク足チャートを作成する。
    
    Args:
        df (pd.DataFrame): テクニカルデータを含むデータフレーム。
        ticker_symbol (str): プロットするシンボル。
        overlays (list): メインチャートに表示するオーバーレイ指標のリスト（例: ['SMA_20', 'BB']）。
        oscillators (list): サブプロットに表示するオシレーターのリスト（例: ['RSI_14', 'MACD']）。
        
    Returns:
        go.Figure: Plotlyの図オブジェクト。
    """
    if df.empty:
        return go.Figure()

    overlays = overlays or []
    oscillators = oscillators or []
    
    # リクエストされた画像（青/赤）に合わせて色を定義
    TV_BG_COLOR = "#131722"
    TV_GRID_COLOR = "#2B2B43"
    TV_TEXT_COLOR = "#D1D4DC"
    TV_UP_COLOR = "#0000FF" # 青
    TV_UP_FILL = "rgba(0, 0, 255, 0.5)"
    TV_DOWN_COLOR = "#FF0000" # 赤
    TV_DOWN_FILL = "rgba(255, 0, 0, 0.5)"

    # 行数を計算:
    # 行1: ローソク足 + オーバーレイ + 出来高（出来高は同じX軸を共有、オーバーレイまたは別）
    # 実際、TradingViewを模倣する場合、通常は出来高をメインチャートの最下部にオーバーレイする。
    # Plotlyでこれをきれいに実装するには、出来高を行1の第2Y軸上の別トレースとして作成し、
    # ドメインを制限して下部20%に留める。
    
    num_rows = 1 + len(oscillators)
    row_heights = [0.6] + [0.4 / len(oscillators)] * len(oscillators) if oscillators else [1.0]
    
    # サブプロットを作成
    fig = make_subplots(
        rows=num_rows, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] * len(oscillators)
    )

    # 1. メインローソク足チャート
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price",
            increasing_line_color=TV_UP_COLOR,
            decreasing_line_color=TV_DOWN_COLOR,
            increasing_fillcolor=TV_UP_COLOR,
            decreasing_fillcolor=TV_DOWN_COLOR,
            line=dict(width=2),  # ローソク足の線の太さを2に増やす
            showlegend=False
        ),
        row=1, col=1, secondary_y=False
    )

    # 2. 出来高オーバーレイ
    if 'Volume' in df.columns:
        colors = [TV_UP_FILL if row['Close'] >= row['Open'] else TV_DOWN_FILL for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name="Volume",
                marker_color=colors,
                showlegend=False,
                hoverinfo='none',
                opacity=0.3
            ),
            row=1, col=1, secondary_y=True
        )

    # 3. オーバーレイ（メインチャート）
    overlay_colors = ['#f5cb42', '#2962FF', '#E91E63', '#00BCD4']
    color_idx = 0
    
    for overlay in overlays:
        if overlay.startswith('SMA_') or overlay.startswith('EMA_'):
            if overlay in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[overlay], mode='lines', name=overlay.replace('_', ' '), 
                               line=dict(color=overlay_colors[color_idx % len(overlay_colors)], width=1.5)),
                    row=1, col=1, secondary_y=False
                )
                color_idx += 1
        elif overlay.startswith('BB_Mid_'):
            # ボリンジャーバンド
            base = overlay.replace('BB_Mid_', '')
            high_col = f'BB_High_{base}'
            low_col = f'BB_Low_{base}'
            if all(col in df.columns for col in [overlay, high_col, low_col]):
                # バンドを追加
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[high_col], mode='lines', name='BB High', 
                               line=dict(color='rgba(41, 98, 255, 0.4)', width=1)),
                    row=1, col=1, secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[low_col], mode='lines', name='BB Low', 
                               line=dict(color='rgba(41, 98, 255, 0.4)', width=1), 
                               fill='tonexty', fillcolor='rgba(41, 98, 255, 0.05)'),
                    row=1, col=1, secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[overlay], mode='lines', name='BB Mid', 
                               line=dict(color='#2962FF', width=1, dash='dot')),
                    row=1, col=1, secondary_y=False
                )

    # 4. オシレーター（サブプロット）
    current_row = 2
    for osc in oscillators:
        if osc.startswith('RSI_'):
            if osc in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[osc], mode='lines', name='RSI', line=dict(color='#7E57C2', width=1.5)),
                    row=current_row, col=1
                )
                # TradingViewスタイルのRSI背景バンドを追加
                fig.add_shape(type="rect",
                    x0=df.index[0], x1=df.index[-1], y0=30, y1=70,
                    fillcolor="rgba(126, 87, 194, 0.1)", line_width=0,
                    layer="below", row=current_row, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,255,255,0.2)", row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="rgba(255,255,255,0.2)", row=current_row, col=1)
                fig.update_yaxes(title_text="RSI", row=current_row, col=1, range=[0, 100], tickvals=[30, 70])
                current_row += 1
                
        elif osc == 'MACD':
            if all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='#2962FF', width=1.5)),
                    row=current_row, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal', line=dict(color='#FF6D00', width=1.5)),
                    row=current_row, col=1
                )
                # ヒストグラムの色
                hist_colors = [TV_UP_COLOR if val >= 0 else TV_DOWN_COLOR for val in df['MACD_Hist']]
                fig.add_trace(
                    go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram', marker_color=hist_colors, opacity=0.8),
                    row=current_row, col=1
                )
                fig.update_yaxes(title_text="MACD", row=current_row, col=1)
                current_row += 1

    # より良いY軸スケーリングロジック
    ymin = df['Low'].min()
    ymax = df['High'].max()
    margin = (ymax - ymin) * 0.05
    ymin -= margin
    ymax += margin

    # TradingViewに合わせてレイアウトをカスタマイズ
    fig.update_layout(
        title=dict(text=f"{ticker_symbol}", font=dict(color=TV_TEXT_COLOR, size=20)),
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        margin=dict(l=10, r=50, t=50, b=20),
        height=600 + (len(oscillators) * 150),
        plot_bgcolor=TV_BG_COLOR,
        paper_bgcolor=TV_BG_COLOR,
        font=dict(color=TV_TEXT_COLOR),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor='rgba(0,0,0,0)'
        ),
        dragmode='pan',
        yaxis=dict(
            range=[ymin, ymax]
        )
    )
    
    # TradingViewのグリッドとクロスヘアに合わせて軸を設定
    fig.update_xaxes(
        type='date', # 日付軸を強制
        showgrid=True, gridwidth=1, gridcolor=TV_GRID_COLOR,
        zeroline=False,
        showspikes=True, spikemode="across", spikesnap="cursor", showline=True, spikedash="dot",
        spikecolor=TV_TEXT_COLOR, spikethickness=1,
        rangebreaks=[
            dict(bounds=["sat", "mon"]) # 週末を非表示
        ]
    )
    # 第1Y軸（価格）を設定 -> 右側を使用
    fig.update_yaxes(
        type='linear', # 線形軸を強制
        showgrid=True, gridwidth=1, gridcolor=TV_GRID_COLOR,
        zeroline=False,
        showspikes=True, spikemode="across", spikesnap="cursor", showline=False, spikedash="dot",
        spikecolor=TV_TEXT_COLOR, spikethickness=1,
        side='right', # TradingViewは価格軸を右側に配置
        row=1, col=1,
        fixedrange=False # ズームを許可
    )
    
    # 統一ホバーを有効化
    fig.update_traces(xaxis="x")
    
    # 第2Y軸（出来高）を設定
    if 'Volume' in df.columns:
        fig.update_yaxes(
            title_text="", 
            showgrid=False, 
            secondary_y=True, 
            showticklabels=False,
            row=1, col=1,
            autorange=True,
            fixedrange=True
        )

    # 最下部のサブプロット以外のX軸目盛りを非表示
    if oscillators:
        fig.update_xaxes(showticklabels=False)
        fig.update_xaxes(showticklabels=True, row=num_rows, col=1)

    return fig

def create_mini_line_chart(df: pd.DataFrame, height: int = 80, width: int = None) -> go.Figure:
    """
    ダッシュボードタイル用のミニマリストなPlotly折れ線チャート（スパークライン）を作成する。
    軸、グリッド、凡例を削除してクリーンな見た目にする。
    
    Args:
        df: 'Close'列を含むデータフレーム。
        height: チャートの高さ（ピクセル）。
        width: チャートの幅（ピクセル、Noneで自動）。
        
    Returns:
        go.Figure: Plotlyのミニ折れ線図。
    """
    if df.empty or 'Close' not in df.columns or len(df) < 2:
        # 空のFigureではなく、最小限のデータでチャートを作成
        empty_fig = go.Figure()
        empty_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 0], mode='lines', line=dict(color='rgba(0,0,0,0)')))
        empty_fig.update_layout(
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',  # 透明な背景（Plotlyでは'rgba(0,0,0,0)'を使用）
            paper_bgcolor='rgba(0,0,0,0)',  # 透明な背景（Plotlyでは'rgba(0,0,0,0)'を使用）
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return empty_fig

    # トレンドに基づいて色を決定（Series になる場合も安全に float へ変換）
    start_raw = df["Close"].iloc[0]
    end_raw = df["Close"].iloc[-1]
    if isinstance(start_raw, pd.Series):
        start_price = float(start_raw.iloc[0])
    else:
        start_price = float(start_raw)
    if isinstance(end_raw, pd.Series):
        end_price = float(end_raw.iloc[0])
    else:
        end_price = float(end_raw)
    
    TV_UP_COLOR = "#089981"
    TV_DOWN_COLOR = "#F23645"
    line_color = TV_UP_COLOR if end_price >= start_price else TV_DOWN_COLOR

    # 塗りつぶし領域のrgba定義（10%の不透明度）
    r = int(line_color[1:3], 16)
    g = int(line_color[3:5], 16)
    b = int(line_color[5:7], 16)
    fill_color = f"rgba({r}, {g}, {b}, 0.1)"

    # Close 列を安全にリスト化（Series / DataFrame 両対応）
    close_col = df["Close"]
    if isinstance(close_col, pd.DataFrame):
        close_series = close_col.iloc[:, 0]
    else:
        close_series = close_col
    y_values = close_series.astype(float).to_list()

    fig = go.Figure(data=[go.Scatter(
        x=list(range(len(y_values))),  # Plotlyが線を伸縮できるように単純な整数範囲を使用
        y=y_values,  # リストに変換して確実にデータを渡す
        mode='lines',
        line=dict(color=line_color, width=2.5),
        fill='tozeroy',
        fillcolor=fill_color,
        hoverinfo='none', # ダッシュボードタイルのホバーを無効化してクリーンに保つ
        showlegend=False,
        connectgaps=True  # データの欠損を補間
    )])

    # Plotlyが線を潰さないように明示的に制限を計算
    ymin = float(min(y_values))
    ymax = float(max(y_values))
    # データが1つの値のみの場合の処理
    if ymax == ymin:
        margin = abs(ymax) * 0.02 if ymax != 0 else 0.01
    else:
        margin = (ymax - ymin) * 0.15  # マージンを少し増やして線が見やすくする
    
    # y軸の範囲を計算（最小値と最大値にマージンを追加）
    y_range_min = ymin - margin
    y_range_max = ymax + margin
    
    # すべての軸、グリッド、ツールバー、レンジスライダー、境界線を削除
    layout_args = dict(
        height=height,
        margin=dict(l=0, r=0, t=2, b=2), # 上下にわずかなマージンを追加
        plot_bgcolor='rgba(0,0,0,0)',  # 透明な背景（Plotlyでは'rgba(0,0,0,0)'を使用）
        paper_bgcolor='rgba(0,0,0,0)',  # 透明な背景（Plotlyでは'rgba(0,0,0,0)'を使用）
        showlegend=False,
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            fixedrange=True, 
            visible=False,
            range=[-0.5, len(df) - 0.5]  # x軸の範囲を明示的に設定
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            fixedrange=True, 
            visible=False, 
            range=[y_range_min, y_range_max]  # y軸の範囲を明示的に設定
        )
    )
    
    if width:
        layout_args['width'] = width

    fig.update_layout(**layout_args)

    return fig

def render_google_candlestick_chart(df: pd.DataFrame, ticker_symbol: str, overlays: list = None) -> str:
    """
    オプションのトレンドラインをオーバーレイしたGoogle Chartsローソク足チャートをレンダリングする。
    Google Chartsは次の形式のデータを期待する: [ 'Date', Low, Open, Close, High ]
    オーバーレイがある場合、ComboChartを使用して列を追加する。
    
    Returns:
        str: チャートを表す生のHTML + JS文字列。
    """
    if df.empty:
        return "<div style='color:white;'>No data available.</div>"

    overlays = overlays or []
    
    # データポイント数に基づいてローソク足の幅を動的に調整
    data_count = len(df)
    if data_count <= 100:
        # データが少ない場合（1日足、1週間足など）は太く表示
        bar_group_width = '95%'
        stroke_width = 3.5
    elif data_count <= 300:
        # データが中程度の場合（4時間足、1日足など）は中程度の太さ
        bar_group_width = '85%'
        stroke_width = 3
    elif data_count <= 800:
        # データが多い場合（1時間足、15分足など）は少し細めだが見やすく
        bar_group_width = '75%'
        stroke_width = 2.5
    else:
        # データが非常に多い場合（5分足など）は細く表示
        bar_group_width = '60%'
        stroke_width = 2
    
    # JSデータ配列を構築する必要がある。
    # 最初の行がヘッダー: ['Day', 'Price (Candles)', 'Overlay1', 'Overlay2'...]
    # ローソク足列はGoogle Chartsによって4つの値として扱われる。
    js_data = []
    
    # ヘッダー
    headers = ["'Date'", "'Low'", "'Open'", "'Close'", "'High'"]
    for o in overlays:
        if o in df.columns:
            headers.append(f"'{o}'")
    js_data.append(f"[{', '.join(headers)}]")
    
    # データ行
    for idx, row in df.iterrows():
        # Google ComboChart for Candlesticks is tricky:
        # The first column is X-axis (string or date).
        # The next 4 columns MUST be the candlestick values: Low, Open, Close, High
        date_str = f"'{idx.strftime('%Y-%m-%d %H:%M')}'"
        low = str(row['Low'])
        open_val = str(row['Open'])
        close_val = str(row['Close'])
        high = str(row['High'])
        
        row_arr = [date_str, low, open_val, close_val, high]
        
        for o in overlays:
            if o in df.columns:
                val = row[o]
                row_arr.append("null" if pd.isna(val) else str(val))
                
        js_data.append(f"[{', '.join(row_arr)}]")
        
    js_data_str = ",\n        ".join(js_data)
    
    # シリーズ設定
    # シリーズ0がローソク足（ドメイン列の後の最初の4つの数値列を意味する）
    # シリーズ1以降が線。
    series_config = "0: {type: 'candlesticks'}"
    for i in range(len(overlays)):
        series_config += f", {i+1}: {{type: 'line'}}"
        
    # HTML/JSを構築
    html = f"""
    <html>
      <head>
        <script type='text/javascript' src='https://www.gstatic.com/charts/loader.js'></script>
        <script type='text/javascript'>
          google.charts.load('current', {{'packages':['corechart']}});
          google.charts.setOnLoadCallback(drawChart);

          function drawChart() {{
            var data = google.visualization.arrayToDataTable([
        {js_data_str}
            ], true); // trueは最初の行がヘッダーではないことを意味するが、ComboChartでシリーズ名が必要な場合は実際にはヘッダー。
            // 待って、ComboChartでのローソク足 + 線の場合:
            // arrayToDataTableの第2引数がtrueの場合、最初の行がデータであり、ヘッダーではないことを意味する。
            // しかし、ヘッダーを提供した。したがって、falseを使用する必要がある。
            
          }}
        </script>
      </head>
      <body>
        <div id='chart_div' style='width: 100%; height: 600px;'></div>
      </body>
    </html>
    """
    
    # ComboChart用にarrayToDataTable呼び出しを修正
    html = f"""
    <html>
      <head>
        <script type='text/javascript' src='https://www.gstatic.com/charts/loader.js'></script>
        <script type='text/javascript'>
          google.charts.load('current', {{'packages':['corechart']}});
          google.charts.setOnLoadCallback(drawChart);

          function drawChart() {{
            var data = new google.visualization.DataTable();
            data.addColumn('string', 'Date');
            data.addColumn('number', 'Low');
            data.addColumn('number', 'Open');
            data.addColumn('number', 'Close');
            data.addColumn('number', 'High');
    """
    
    # 動的オーバーレイ列を追加
    for o in overlays:
        if o in df.columns:
            html += f"        data.addColumn('number', '{o}');\n"
            
    html += "        data.addRows([\n"
    
    data_rows = []
    for idx, row in df.iterrows():
        date_str = f"'{idx.strftime('%Y-%m-%d %H:%M')}'"
        low = str(row['Low'])
        open_val = str(row['Open'])
        close_val = str(row['Close'])
        high = str(row['High'])
        
        r_arr = [date_str, low, open_val, close_val, high]
        for o in overlays:
            if o in df.columns:
                val = row[o]
                r_arr.append("null" if pd.isna(val) else str(val))
        data_rows.append(f"          [{', '.join(r_arr)}]")
        
    html += ",\n".join(data_rows)
    html += "\n        ]);\n"
    
    html += f"""
            var options = {{
              legend: 'none',
              backgroundColor: '#161b22',
              chartArea: {{left: 60, top: 20, width: '88%', height: '78%'}},
              hAxis: {{
                textStyle: {{color: '#8b949e', fontSize: 11}},
                gridlines: {{color: '#30363d'}},
                baselineColor: '#30363d'
              }},
              vAxis: {{
                textStyle: {{color: '#8b949e', fontSize: 11}},
                gridlines: {{color: '#30363d'}},
                baselineColor: '#30363d'
              }},
              focusTarget: 'category',
              candlestick: {{
                fallingColor: {{ strokeWidth: {stroke_width}, stroke: '#f43f5e', fill: '#f43f5e' }},
                risingColor:  {{ strokeWidth: {stroke_width}, stroke: '#3b82f6', fill: '#3b82f6' }},
                hollowIsRising: false,
              }},
              bar: {{
                groupWidth: '{bar_group_width}'
              }},
              explorer: {{
                actions: ['dragToZoom', 'rightClickToReset'],
                axis: 'horizontal',
                keepInBounds: true,
                maxZoomIn: 0.1,
                maxZoomOut: 10
              }},
              seriesType: 'candlesticks',
              series: {{
    """
    # ローソク足はseriesTypeで処理される。追加の列が線。
    # 注意: ComboChartでは、ローソク足 + 線のみを使用する場合、
    # 4列のデータ定義がサポートされる。
    # 実際、ComboChartを使用したGoogle Chartsローソク足:
    # 最初のシリーズ（インデックス0）がローソク足で、列1,2,3,4を消費する。
    # 次のシリーズ（インデックス1）が列5にマッピングされる。
    
    for i in range(len(overlays)):
        html += f"            {i+1}: {{type: 'line', lineWidth: 1.5}},\n"
        
    html += f"""
              }}
            }};

            var chart = new google.visualization.ComboChart(document.getElementById('chart_div'));
            var totalRows = data.getNumberOfRows();
            var visibleRows = Math.min(300, totalRows); // 初期表示は最大300件
            var startRow = Math.max(0, totalRows - visibleRows);
            
            // データが多い場合は初期表示範囲を制限
            var view = null;
            if (totalRows > 300) {{
              view = new google.visualization.DataView(data);
              view.setRows(startRow, totalRows - 1);
              chart.draw(view, options);
            }} else {{
              chart.draw(data, options);
            }}
            
            // スクロールコントロールを追加（データが多い場合のみ）
            if (totalRows > 300) {{
              var scrollContainer = document.getElementById('scroll_container');
              var scrollBar = document.getElementById('scroll_bar');
              var scrollInfo = document.getElementById('scroll_info');
              
              // スクロールバーの設定
              var maxScroll = totalRows - visibleRows;
              scrollBar.max = maxScroll;
              scrollBar.value = startRow;
              
              // 表示範囲の更新
              function updateView() {{
                var currentStart = parseInt(scrollBar.value);
                var currentEnd = Math.min(currentStart + visibleRows, totalRows);
                
                view = new google.visualization.DataView(data);
                view.setRows(currentStart, currentEnd - 1);
                chart.draw(view, options);
                
                scrollInfo.textContent = `表示中: ${{currentStart + 1}} - ${{currentEnd}} / ${{totalRows}}件`;
              }}
              
              scrollBar.addEventListener('input', updateView);
              
              // 初期表示情報
              scrollInfo.textContent = `表示中: ${{startRow + 1}} - ${{Math.min(startRow + visibleRows, totalRows)}} / ${{totalRows}}件`;
            }}
          }}
        </script>
      </head>
      <body style="margin:0; padding:0; background-color:#161b22;">
        <div id='chart_div' style='width: 100%; height: 600px;'></div>
        <div id='scroll_container' style='padding:15px; background-color:#161b22; border-top:1px solid #30363d;'>
          <div id='scroll_info' style='color:#8b949e; font-size:12px; margin-bottom:8px; text-align:center;'></div>
          <input type='range' id='scroll_bar' min='0' max='1000' value='0' 
                 style='width:100%; height:8px; background:#30363d; border-radius:4px; outline:none; cursor:pointer;'
                 oninput='this.style.background = `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${{(this.value / this.max) * 100}}%, #30363d ${{(this.value / this.max) * 100}}%, #30363d 100%)`'>
          <div style='color:#8b949e; font-size:11px; margin-top:8px; text-align:center;'>
            💡 スクロールバーをドラッグしてチャートを移動
          </div>
        </div>
      </body>
    </html>
    """
    
    return html
