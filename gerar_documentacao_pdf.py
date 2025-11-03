"""
Script para gerar documentação PDF detalhada do app.py
"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted, Table, TableStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("AVISO: reportlab nao esta instalado. Instalando...")
    os.system(f"{sys.executable} -m pip install reportlab -q")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted, Table, TableStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
        from reportlab.lib.colors import HexColor, black, white
        from reportlab.lib import colors
        REPORTLAB_AVAILABLE = True
    except:
        print("ERRO: Nao foi possivel instalar reportlab. Use: pip install reportlab")
        sys.exit(1)


def read_app_py():
    """Lê o arquivo app.py."""
    app_py_path = Path("app.py")
    if not app_py_path.exists():
        # Tentar encontrar em outro local
        app_py_path = Path(__file__).parent / "app.py"
    
    if not app_py_path.exists():
        raise FileNotFoundError("app.py nao encontrado")
    
    with open(app_py_path, 'r', encoding='utf-8') as f:
        return f.read(), app_py_path


def extract_all_methods(code_content):
    """Extrai todos os métodos e funções do código."""
    methods = []
    lines = code_content.split('\n')
    
    current_method = None
    in_method = False
    method_indent = 0
    method_lines = []
    
    for i, line in enumerate(lines, start=1):
        # Detectar início de função/método
        func_match = re.match(r'^(\s*)(def|class)\s+(\w+)\s*\(', line)
        if func_match:
            # Salvar método anterior se existir
            if current_method and method_lines:
                methods.append({
                    'name': current_method['name'],
                    'type': current_method['type'],
                    'line_start': current_method['line_start'],
                    'line_end': i - 1,
                    'code': '\n'.join(method_lines),
                    'signature': current_method['signature']
                })
            
            # Novo método
            indent = len(func_match.group(1))
            method_type = func_match.group(2)
            method_name = func_match.group(3)
            
            current_method = {
                'name': method_name,
                'type': method_type,
                'line_start': i,
                'signature': line.strip()
            }
            method_indent = indent
            method_lines = [line]
            in_method = True
            continue
        
        # Coletar linhas do método atual
        if in_method and current_method:
            current_indent = len(line) - len(line.lstrip()) if line.strip() else method_indent
            
            # Parar se encontrou outro método/classe no mesmo nível ou acima
            if line.strip() and current_indent <= method_indent:
                if re.match(r'^\s*(def|class)\s+', line) and len(line) - len(line.lstrip()) <= method_indent:
                    # Finalizar método anterior
                    methods.append({
                        'name': current_method['name'],
                        'type': current_method['type'],
                        'line_start': current_method['line_start'],
                        'line_end': i - 1,
                        'code': '\n'.join(method_lines),
                        'signature': current_method['signature']
                    })
                    # Começar novo método
                    func_match = re.match(r'^(\s*)(def|class)\s+(\w+)\s*\(', line)
                    if func_match:
                        indent = len(func_match.group(1))
                        method_type = func_match.group(2)
                        method_name = func_match.group(3)
                        current_method = {
                            'name': method_name,
                            'type': method_type,
                            'line_start': i,
                            'signature': line.strip()
                        }
                        method_indent = indent
                        method_lines = [line]
                    continue
            
            method_lines.append(line)
    
    # Adicionar último método
    if current_method and method_lines:
        methods.append({
            'name': current_method['name'],
            'type': current_method['type'],
            'line_start': current_method['line_start'],
            'line_end': len(lines),
            'code': '\n'.join(method_lines),
            'signature': current_method['signature']
        })
    
    return methods


def extract_docstring(code_block):
    """Extrai docstring de um bloco de código."""
    docstring_match = re.search(r'"""(.*?)"""', code_block, re.DOTALL)
    if docstring_match:
        return docstring_match.group(1).strip()
    return None


def analyze_method(method_info):
    """Analisa um método e retorna informações detalhadas."""
    code = method_info['code']
    docstring = extract_docstring(code)
    
    # Contar linhas
    lines = [l for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
    line_count = len(lines)
    
    # Identificar imports/usos
    uses = []
    if 'self.camera' in code:
        uses.append("BaslerCamera")
    if 'self.detector' in code:
        uses.append("YOLODetector")
    if 'self.ui' in code:
        uses.append("YOLODetectionUI")
    if 'self.config' in code:
        uses.append("ConfigManager")
    if 'logger.' in code:
        uses.append("Logging")
    if 'threading.' in code or 'Thread(' in code:
        uses.append("Threading")
    if 'cv2.' in code or 'VideoWriter' in code:
        uses.append("OpenCV")
    if 'yaml.' in code:
        uses.append("YAML")
    if 'torch.' in code:
        uses.append("PyTorch")
    
    # Identificar operações
    operations = []
    if 'process_frame' in code:
        operations.append("Processamento de Frame")
    if 'get_frame' in code:
        operations.append("Captura de Frame")
    if 'VideoWriter' in code:
        operations.append("Gravação de Vídeo")
    if 'update_frame' in code or 'update_stats' in code:
        operations.append("Atualização de UI")
    if 'save' in code.lower() and ('config' in code or 'parameter' in code):
        operations.append("Persistência de Dados")
    if 'load' in code.lower() and ('config' in code or 'parameter' in code):
        operations.append("Carregamento de Dados")
    
    # Identificar parâmetros
    params_match = re.search(r'\(([^)]*)\)', method_info['signature'])
    params = []
    if params_match:
        param_str = params_match.group(1)
        if param_str.strip():
            params = [p.strip().split('=')[0].strip() for p in param_str.split(',')]
    
    return {
        'docstring': docstring,
        'line_count': line_count,
        'uses': uses,
        'operations': operations,
        'params': params,
        'code_preview': '\n'.join(code.split('\n')[:30])  # Primeiras 30 linhas
    }


def generate_pdf():
    """Gera PDF com documentação detalhada do app.py."""
    print("Gerando documentacao PDF do app.py...")
    
    # Ler app.py
    try:
        code_content, app_py_path = read_app_py()
        print(f"[OK] Arquivo lido: {app_py_path}")
    except Exception as e:
        print(f"[ERRO] Erro ao ler app.py: {e}")
        return False
    
    # Criar PDF
    output_path = Path("Documentacao_App.py.pdf")
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=30
    )
    
    # Estilos
    styles = getSampleStyleSheet()
    
    # Estilos customizados
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=HexColor('#1B5E20'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=HexColor('#2E7D32'),
        spaceAfter=10,
        spaceBefore=15,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=13,
        textColor=HexColor('#388E3C'),
        spaceAfter=8,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Code'],
        fontSize=8,
        fontName='Courier',
        leading=10,
        leftIndent=10,
        rightIndent=10,
        spaceAfter=8,
        backColor=HexColor('#F5F5F5'),
        borderColor=HexColor('#CCCCCC'),
        borderWidth=1,
        borderPadding=5
    )
    
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=12,
        alignment=TA_JUSTIFY,
        spaceAfter=8
    )
    
    # Story (conteúdo do PDF)
    story = []
    
    # Título
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Documentação Técnica Completa", title_style))
    story.append(Paragraph("app.py - Sistema de Detecção YOLO", styles['Heading2']))
    story.append(Paragraph("Sistema de Detecção YOLO com Câmera Basler USB3 Vision", styles['Normal']))
    story.append(Paragraph(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(PageBreak())
    
    # Extrair todos os métodos
    all_methods = extract_all_methods(code_content)
    
    # Separar por tipo
    classes = [m for m in all_methods if m['type'] == 'class']
    functions = [m for m in all_methods if m['type'] == 'def' and not m['name'].startswith('_')]
    methods = [m for m in all_methods if m['type'] == 'def' and m['name'].startswith('_')]
    public_methods = [m for m in all_methods if m['type'] == 'def' and not m['name'].startswith('_') and m['name'] != 'main']
    
    # 1. Visão Geral
    story.append(Paragraph("1. Visão Geral do Sistema", heading_style))
    
    overview_text = """
    O arquivo <b>app.py</b> é o módulo principal do Sistema de Detecção YOLO com câmera Basler USB3 Vision.
    Este sistema integra captura de vídeo em tempo real, inferência usando modelos YOLO e uma interface
    gráfica profissional para controle e visualização.
    
    <b>Componentes Principais:</b><br/>
    • <b>YOLODetectionApp:</b> Classe principal que orquestra todo o sistema<br/>
    • <b>BaslerCamera:</b> Interface com câmera Basler USB3 Vision<br/>
    • <b>YOLODetector:</b> Sistema de detecção multi-modelo YOLO<br/>
    • <b>YOLODetectionUI:</b> Interface gráfica Tkinter profissional<br/>
    • <b>ConfigManager:</b> Gerenciamento de configurações persistentes<br/>
    
    <b>Funcionalidades:</b><br/>
    • Captura de vídeo em tempo real<br/>
    • Detecção multi-modelo (ROI, Smudge, Símbolos, BlackDot)<br/>
    • Interface gráfica com controles avançados<br/>
    • Gravação de vídeo com múltiplos codecs<br/>
    • Sistema de estatísticas e análise de transfers<br/>
    • Configuração persistente de parâmetros<br/>
    """
    
    story.append(Paragraph(overview_text, normal_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(PageBreak())
    
    # 2. Imports e Dependências
    story.append(Paragraph("2. Imports e Dependências", heading_style))
    
    imports_text = """
    <b>Bibliotecas Padrão Python:</b><br/>
    • os, sys: Operações do sistema operacional<br/>
    • logging: Sistema de logging para registro de eventos<br/>
    • threading: Multithreading para processamento paralelo<br/>
    • time: Operações com tempo<br/>
    • queue: Filas para comunicação entre threads<br/>
    • datetime: Manipulação de datas e horários<br/>
    • pathlib: Manipulação de caminhos de arquivos<br/>
    • typing: Type hints para melhor documentação<br/>
    
    <b>Bibliotecas Externas:</b><br/>
    • <b>yaml:</b> Manipulação de arquivos YAML de configuração<br/>
    • <b>cv2 (OpenCV):</b> Processamento de imagem e vídeo<br/>
    • <b>numpy:</b> Operações numéricas e arrays<br/>
    • <b>torch (PyTorch):</b> Framework de deep learning para YOLO<br/>
    
    <b>Módulos Personalizados:</b><br/>
    • <b>BaslerCamera:</b> Interface com câmera Basler USB3 Vision<br/>
    • <b>YOLODetector:</b> Sistema de detecção YOLO multi-modelo<br/>
    • <b>YOLODetectionUI:</b> Interface gráfica Tkinter<br/>
    • <b>ConfigManager:</b> Gerenciamento de configurações persistentes<br/>
    """
    
    story.append(Paragraph(imports_text, normal_style))
    story.append(PageBreak())
    
    # 3. Função setup_logging()
    story.append(Paragraph("3. Função setup_logging()", heading_style))
    
    setup_logging_code = [m for m in functions if m['name'] == 'setup_logging']
    if setup_logging_code:
        method = setup_logging_code[0]
        analysis = analyze_method(method)
        
        description = f"""
        <b>Propósito:</b> Configura o sistema de logging da aplicação.<br/><br/>
        
        <b>Descrição:</b> {analysis['docstring'] or 'Configura sistema de logging para arquivo e console'}<br/><br/>
        
        <b>Funcionalidade:</b><br/>
        • Cria diretório 'logs' se não existir<br/>
        • Gera arquivo de log com timestamp único<br/>
        • Configura handlers para arquivo e console<br/>
        • Retorna logger configurado para uso no sistema<br/><br/>
        
        <b>Retorno:</b> Objeto logger configurado<br/>
        <b>Linhas de código:</b> {analysis['line_count']}<br/>
        """
        
        story.append(Paragraph(description, normal_style))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("<b>Código:</b>", subheading_style))
        story.append(Preformatted(method['code'], code_style))
        story.append(PageBreak())
    
    # 4. Classe YOLODetectionApp
    story.append(Paragraph("4. Classe YOLODetectionApp", heading_style))
    
    class_info = [c for c in classes if c['name'] == 'YOLODetectionApp']
    if class_info:
        class_obj = class_info[0]
        class_docstring = extract_docstring(class_obj['code'])
        
        class_desc = f"""
        <b>Descrição:</b> {class_docstring or 'Aplicação principal de detecção YOLO'}<br/><br/>
        
        <b>Responsabilidades:</b><br/>
        • Orquestrar inicialização de todos os componentes<br/>
        • Gerenciar ciclo de vida da aplicação<br/>
        • Coordenar comunicação entre câmera, detector e UI<br/>
        • Gerenciar threads de processamento<br/>
        • Controlar gravação de vídeo<br/>
        • Salvar/carregar configurações<br/><br/>
        
        <b>Atributos Principais:</b><br/>
        • logger: Sistema de logging<br/>
        • config: Dicionário de configurações<br/>
        • camera: Instância BaslerCamera<br/>
        • detector: Instância YOLODetector<br/>
        • ui: Instância YOLODetectionUI<br/>
        • running: Flag de controle de execução<br/>
        • recording: Flag de controle de gravação<br/>
        • video_writer: Objeto VideoWriter do OpenCV<br/>
        """
        
        story.append(Paragraph(class_desc, normal_style))
        story.append(PageBreak())
    
    # Agrupar métodos por categoria
    init_methods = [m for m in methods if 'init' in m['name']]
    callback_methods = [m for m in methods if 'on_ui' in m['name']]
    recording_methods = [m for m in methods if 'recording' in m['name']]
    config_methods = [m for m in methods if 'config' in m['name'] or 'cuda' in m['name'] or 'optimize' in m['name']]
    control_methods = [m for m in public_methods if m['name'] in ['start', 'stop', 'run']]
    utility_methods = [m for m in methods if m['name'] not in init_methods + callback_methods + recording_methods + config_methods and m['name'] != '__init__']
    
    # 5. Método __init__
    story.append(Paragraph("5. Método __init__()", heading_style))
    
    init_method = [m for m in methods if m['name'] == '__init__']
    if init_method:
        method = init_method[0]
        analysis = analyze_method(method)
        
        desc = f"""
        <b>Propósito:</b> Inicializa a aplicação e todos os seus componentes.<br/><br/>
        
        <b>Descrição:</b> {analysis['docstring'] or 'Inicializa a aplicação'}<br/><br/>
        
        <b>Parâmetros:</b> {', '.join(analysis['params']) if analysis['params'] else 'config_path (opcional)'}<br/><br/>
        
        <b>Processo de Inicialização:</b><br/>
        1. Configura sistema de logging<br/>
        2. Carrega configuração YAML<br/>
        3. Verifica disponibilidade CUDA<br/>
        4. Aplica otimizações para RTX 3050 (se detectada)<br/>
        5. Inicializa componentes (câmera, detector, UI serão inicializados em run())<br/><br/>
        
        <b>Componentes Utilizados:</b> {', '.join(analysis['uses']) if analysis['uses'] else 'Nenhum específico'}<br/>
        <b>Linhas de código:</b> {analysis['line_count']}<br/>
        """
        
        story.append(Paragraph(desc, normal_style))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("<b>Código:</b>", subheading_style))
        story.append(Preformatted(method['code'], code_style))
        story.append(PageBreak())
    
    # 6. Métodos de Configuração
    story.append(Paragraph("6. Métodos de Configuração", heading_style))
    
    for i, method in enumerate(config_methods[:5], 1):  # Primeiros 5
        analysis = analyze_method(method)
        
        story.append(Paragraph(f"6.{i}. {method['name']}()", subheading_style))
        
        desc = f"""
        <b>Linhas:</b> {method['line_start']}-{method['line_end']} | <b>Código:</b> {analysis['line_count']} linhas<br/>
        <b>Descrição:</b> {analysis['docstring'] or 'Método de configuração'}<br/>
        <b>Usa:</b> {', '.join(analysis['uses']) if analysis['uses'] else 'Nenhum'}<br/>
        """
        
        story.append(Paragraph(desc, normal_style))
        story.append(Preformatted(method['code'][:500] + ('...' if len(method['code']) > 500 else ''), code_style))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(PageBreak())
    
    # 7. Métodos de Inicialização
    story.append(Paragraph("7. Métodos de Inicialização", heading_style))
    
    for i, method in enumerate(init_methods, 1):
        analysis = analyze_method(method)
        
        story.append(Paragraph(f"7.{i}. {method['name']}()", subheading_style))
        
        desc = f"""
        <b>Linhas:</b> {method['line_start']}-{method['line_end']}<br/>
        <b>Descrição:</b> {analysis['docstring'] or 'Método de inicialização'}<br/>
        <b>Operações:</b> {', '.join(analysis['operations']) if analysis['operations'] else 'Inicialização geral'}<br/>
        """
        
        story.append(Paragraph(desc, normal_style))
        story.append(Preformatted(method['code'][:800] + ('...' if len(method['code']) > 800 else ''), code_style))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(PageBreak())
    
    # 8. Métodos de Callback
    story.append(Paragraph("8. Métodos de Callback da UI", heading_style))
    
    story.append(Paragraph("""
    Estes métodos são chamados pela interface gráfica quando o usuário interage com os controles.
    Eles fazem a ponte entre a UI e os componentes do sistema (câmera, detector, etc).
    """, normal_style))
    
    callback_table_data = [['Método', 'Linhas', 'Descrição']]
    for method in callback_methods[:10]:
        analysis = analyze_method(method)
        callback_table_data.append([
            method['name'],
            f"{method['line_start']}-{method['line_end']}",
            (analysis['docstring'] or 'Callback')[:50] + '...' if analysis['docstring'] and len(analysis['docstring']) > 50 else (analysis['docstring'] or 'Callback')
        ])
    
    callback_table = Table(callback_table_data, colWidths=[2.5*inch, 1*inch, 3*inch])
    callback_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E7D32')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#F5F5F5')),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#CCCCCC')),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    
    story.append(callback_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Detalhar principais callbacks
    main_callbacks = ['_on_ui_start', '_on_ui_stop', '_on_ui_record_toggle', '_on_ui_threshold_change']
    for method_name in main_callbacks:
        method = [m for m in callback_methods if m['name'] == method_name]
        if method:
            method = method[0]
            analysis = analyze_method(method)
            story.append(Paragraph(f"8.{main_callbacks.index(method_name) + 1}. {method['name']}()", subheading_style))
            story.append(Paragraph(f"<b>Descrição:</b> {analysis['docstring'] or 'Callback da UI'}", normal_style))
            story.append(Preformatted(method['code'][:600] + ('...' if len(method['code']) > 600 else ''), code_style))
            story.append(Spacer(1, 0.1*inch))
    
    story.append(PageBreak())
    
    # 9. Métodos de Controle
    story.append(Paragraph("9. Métodos de Controle", heading_style))
    
    for method in control_methods:
        analysis = analyze_method(method)
        story.append(Paragraph(f"9.{control_methods.index(method) + 1}. {method['name']}()", subheading_style))
        story.append(Paragraph(f"<b>Descrição:</b> {analysis['docstring'] or 'Método de controle'}", normal_style))
        story.append(Paragraph(f"<b>Operações:</b> {', '.join(analysis['operations']) if analysis['operations'] else 'Controle geral'}", normal_style))
        story.append(Preformatted(method['code'][:1000] + ('...' if len(method['code']) > 1000 else ''), code_style))
        story.append(Spacer(1, 0.2*inch))
    
    story.append(PageBreak())
    
    # 10. Métodos de Gravação
    story.append(Paragraph("10. Métodos de Gravação", heading_style))
    
    for method in recording_methods:
        analysis = analyze_method(method)
        story.append(Paragraph(f"10.{recording_methods.index(method) + 1}. {method['name']}()", subheading_style))
        story.append(Paragraph(f"<b>Descrição:</b> {analysis['docstring'] or 'Método de gravação'}", normal_style))
        story.append(Preformatted(method['code'][:1000] + ('...' if len(method['code']) > 1000 else ''), code_style))
        story.append(Spacer(1, 0.2*inch))
    
    story.append(PageBreak())
    
    # 11. Métodos de Utilidade
    story.append(Paragraph("11. Métodos de Utilidade", heading_style))
    
    utility_table_data = [['Método', 'Linhas', 'Usa', 'Operações']]
    for method in utility_methods[:15]:
        analysis = analyze_method(method)
        utility_table_data.append([
            method['name'],
            f"{method['line_start']}-{method['line_end']}",
            ', '.join(analysis['uses'][:2]) if analysis['uses'] else '-',
            ', '.join(analysis['operations'][:2]) if analysis['operations'] else '-'
        ])
    
    utility_table = Table(utility_table_data, colWidths=[2*inch, 1*inch, 1.5*inch, 2.5*inch])
    utility_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#388E3C')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#F5F5F5')),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#CCCCCC')),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    
    story.append(utility_table)
    story.append(PageBreak())
    
    # 12. Função main()
    story.append(Paragraph("12. Função main()", heading_style))
    
    main_func = [m for m in functions if m['name'] == 'main']
    if main_func:
        method = main_func[0]
        analysis = analyze_method(method)
        
        desc = f"""
        <b>Propósito:</b> Ponto de entrada principal da aplicação.<br/><br/>
        
        <b>Descrição:</b> {analysis['docstring'] or 'Função principal'}<br/><br/>
        
        <b>Fluxo de Execução:</b><br/>
        1. Configura variáveis de ambiente<br/>
        2. Cria diretórios necessários (logs, recordings, models)<br/>
        3. Cria instância de YOLODetectionApp<br/>
        4. Executa app.run()<br/>
        5. Trata exceções e finaliza apropriadamente<br/><br/>
        
        <b>Tratamento de Erros:</b><br/>
        • KeyboardInterrupt: Finalização limpa<br/>
        • Exceções gerais: Exibe traceback completo<br/>
        """
        
        story.append(Paragraph(desc, normal_style))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("<b>Código Completo:</b>", subheading_style))
        story.append(Preformatted(method['code'], code_style))
        story.append(PageBreak())
    
    # 13. Resumo e Estatísticas
    story.append(Paragraph("13. Resumo e Estatísticas", heading_style))
    
    total_lines = len(code_content.split('\n'))
    total_methods = len(all_methods)
    total_classes = len(classes)
    total_functions = len(functions)
    
    stats_text = f"""
    <b>Estatísticas do Código:</b><br/>
    • Total de linhas: {total_lines}<br/>
    • Classes: {total_classes}<br/>
    • Funções: {total_functions}<br/>
    • Métodos: {total_methods - total_functions - total_classes}<br/>
    • Métodos públicos: {len(public_methods)}<br/>
    • Métodos privados: {len(methods)}<br/><br/>
    
    <b>Estrutura por Categoria:</b><br/>
    • Métodos de inicialização: {len(init_methods)}<br/>
    • Métodos de callback: {len(callback_methods)}<br/>
    • Métodos de gravação: {len(recording_methods)}<br/>
    • Métodos de configuração: {len(config_methods)}<br/>
    • Métodos de controle: {len(control_methods)}<br/>
    • Métodos de utilidade: {len(utility_methods)}<br/><br/>
    
    <b>Dependências Principais:</b><br/>
    • PyTorch (CUDA recomendado)<br/>
    • OpenCV (cv2)<br/>
    • Ultralytics YOLO<br/>
    • Basler Pylon SDK (pypylon)<br/>
    • Tkinter (interface gráfica)<br/>
    • PyYAML (configurações)<br/>
    """
    
    story.append(Paragraph(stats_text, normal_style))
    story.append(PageBreak())
    
    # 14. Código Completo (últimas páginas)
    story.append(Paragraph("14. Código Completo (app.py)", heading_style))
    story.append(Paragraph("Código-fonte completo para referência:", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Dividir código em chunks menores
    code_lines = code_content.split('\n')
    chunk_size = 50
    total_chunks = (len(code_lines) + chunk_size - 1) // chunk_size
    
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(code_lines))
        chunk = '\n'.join(code_lines[start_idx:end_idx])
        
        story.append(Paragraph(f"Linhas {start_idx + 1} a {end_idx}:", styles['Normal']))
        story.append(Preformatted(chunk, code_style))
        story.append(Spacer(1, 0.1*inch))
    
    # Gerar PDF
    try:
        doc.build(story)
        print(f"[OK] PDF gerado com sucesso: {output_path.absolute()}")
        return True
    except Exception as e:
        print(f"[ERRO] Erro ao gerar PDF: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = generate_pdf()
    if success:
        print("\n[OK] Documentacao PDF gerada com sucesso!")
        print(f"Arquivo: {Path('Documentacao_App.py.pdf').absolute()}")
    else:
        print("\n[ERRO] Falha ao gerar documentacao PDF")
        sys.exit(1)
