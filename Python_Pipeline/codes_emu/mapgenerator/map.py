import sys
from PyQt6.QtWidgets import QApplication, QWidget, QCheckBox, QLabel, QLineEdit, QComboBox, QPushButton, QTextEdit, QFileDialog, QTabWidget, QGroupBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class MapfileGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Add the following dictionary to keep track of the number of named channels per port
        self.num_named_channels_per_port = {}
        self.generated_channels = []
        self.channel_names = {}
        self.action_history = []

    def initUI(self):
        self.setWindowTitle('Mapfile Generator')
        self.setGeometry(100, 100, 491, 752)
        self.tab_widget = QTabWidget(self)
        self.tab_widget.setGeometry(0, 0, 491, 753)

        # Primer box 
        self.group_box1 = QGroupBox('Generate Channels', self)
        self.group_box1.setGeometry(10, 10, 250, 175)

        self.label_entry = QLabel('Ports:', self)
        self.label_entry.move(20, 45)
        self.combo_entry = QComboBox(self)
        self.combo_entry.addItems(['A', 'B', 'C','D'])
        self.combo_entry.move(70, 45)
        self.combo_entry.currentIndexChanged.connect(self.update_ports_input)

        self.label_ports = QLabel('Leads:', self)
        self.label_ports.move(20, 85)
        self.line_ports = QLineEdit(self)
        self.line_ports.setGeometry(70, 85, 40, 25)

        self.generate_18_channels_checkbox = QCheckBox('18 channels per lead', self.group_box1)
        self.generate_18_channels_checkbox.move(110, 35)

        font = self.generate_18_channels_checkbox.font()
        font.setPointSize(8)
        self.generate_18_channels_checkbox.setFont(font)

        self.generate_button = QPushButton('Generate', self)
        self.generate_button.setGeometry(120, 85, 120, 25)
        self.generate_button.clicked.connect(self.generate_mapfile)

        self.mapfile_text = QTextEdit(self)
        self.mapfile_text.setGeometry(20, 125, 230, 50)
        self.mapfile_text.setReadOnly(True)

        self.generated_content = {}
        self.num_ports = {}

        # Segundo box 
        self.group_box2 = QGroupBox('Name Channels', self)
        self.group_box2.setGeometry(10, 194, 250, 357)

        self.label_channel_name = QLabel('Bundle name:', self)
        self.label_channel_name.move(20, 227)
        self.line_channel_name = QLineEdit(self)
        self.line_channel_name.setGeometry(120, 227, 75, 25)

        self.label_num_channels = QLabel('Number of channels:', self)
        self.label_num_channels.move(20, 267)
        self.line_num_channels = QLineEdit(self)
        self.line_num_channels.setGeometry(170, 267, 40, 25)

        self.name_channels_button = QPushButton('Assign names', self)
        self.name_channels_button.move(30, 307)
        self.name_channels_button.clicked.connect(self.name_channels)

        self.next_port_button = QPushButton('Next lead', self)
        self.next_port_button.move(160, 307)
        self.next_port_button.clicked.connect(self.next_port)

        self.names_preview = QTextEdit(self)
        self.names_preview.setGeometry(20, 345, 230, 194)
        self.names_preview.setPlaceholderText("The names will be assigned to the port that is selected in the generate channels box.")
        self.names_preview.setReadOnly(True)

        # Tercer box 
        self.group_box3 = QGroupBox('Modify Channels', self)
        self.group_box3.setGeometry(10, 560, 250, 179)

        self.label_modify_channel = QLabel('Modify channel:', self)
        self.label_modify_channel.move(20, 597)
        self.line_modify_channel = QLineEdit(self)
        self.line_modify_channel.setGeometry(140, 597, 75, 25)
        self.line_modify_channel.setPlaceholderText("e.g. LA01")

        # Add the dropdown list to select the suffix option
        self.label_suffix_option = QLabel('Suffix:', self)
        self.label_suffix_option.move(20, 637)
        self.combo_suffix_option = QComboBox(self)
        self.combo_suffix_option.addItems(['-Z', '-COM'])
        self.combo_suffix_option.move(70, 637)

        # Button to apply channel modification
        self.modify_channel_button = QPushButton('Add suffix', self)
        self.modify_channel_button.move(160, 637)
        self.modify_channel_button.clicked.connect(self.modify_channel_name)

        self.modify_preview = QTextEdit(self)
        self.modify_preview.setGeometry(20, 677, 230, 50)
        self.modify_preview.setReadOnly(True)

        # Cuarta box
        self.group_box1 = QGroupBox('Preview', self)
        self.group_box1.setGeometry(270, 10, 210, 610)

        self.generated_channels_text = QTextEdit(self)
        self.generated_channels_text.setGeometry(280, 45, 190, 562)
        self.generated_channels_text.setReadOnly(True)

        self.load_mapfile_button = QPushButton('Load Mapfile', self)
        self.load_mapfile_button.setGeometry(285, 635, 180, 25)
        self.load_mapfile_button.clicked.connect(self.load_mapfile)

        self.save_button = QPushButton('Save file', self)
        self.save_button.setGeometry(365, 670, 100, 55)
        self.save_button.clicked.connect(self.save_mapfile)

        self.reset_button = QPushButton('Clear all', self)
        self.reset_button.setGeometry(285, 700, 70, 25)
        self.reset_button.clicked.connect(self.reset_generator)

        self.undo_button = QPushButton('Undo', self)
        self.undo_button.setGeometry(285, 670, 70, 25)
        self.undo_button.clicked.connect(self.undo_last_action)

    def reset_generator(self):
        self.generated_content = {}
        self.num_ports = {}
        self.channel_names = {}
        self.num_named_channels_per_port = {}
        self.generated_channels = []
        self.mapfile_text.clear()
        self.names_preview.clear()
        self.generated_channels_text.clear()
        self.modify_preview.clear()

    def update_ports_input(self):
        entry = self.combo_entry.currentText()
        if entry in self.num_ports:
            self.line_ports.setText(str(self.num_ports[entry]))
        else:
            self.line_ports.setText("")

    def generate_mapfile(self):
        entry = self.combo_entry.currentText()
        ports = int(self.line_ports.text())
        if self.generate_18_channels_checkbox.isChecked():
            channels_per_port = 18
        else:
            channels_per_port = 32

        if entry not in self.generated_content:
            self.generated_content[entry] = []

        if entry in self.num_ports:
            existing_ports = self.num_ports[entry]
        else:
            existing_ports = 0

        if existing_ports >= ports:
            self.update_mapfile_text(f"The {ports} leads have been previously generated for port {entry}.")
            return

        generated_channels_for_entry = []

        for port in range(existing_ports + 1, ports + 1):
            port_content = []
            for channel in range(1, channels_per_port + 1):
                line = f"1.{entry}.{port}.{channel:03d};"
                port_content.append((line, ""))
                generated_channels_for_entry.append(line)  # Add the channel to the list
            self.generated_content[entry].append(port_content)

        self.generated_channels.extend(generated_channels_for_entry)  # Extends the list of generated channels
        self.update_generated_channels_text()
        self.num_ports[entry] = ports
        message = f"Generated {ports - existing_ports} additional leads with {channels_per_port} channels each for port {entry}."
        self.update_mapfile_text(message)

        action_data = {"entry": entry}
        self.record_action(("generate", action_data))


    def record_action(self, action_data):
        self.action_history.append(action_data)

    def undo_last_action(self):
        if self.action_history:
            last_action = self.action_history.pop()
            action_type, data = last_action

            if action_type == "generate":
                entry = data['entry']
                self.undo_generate_action(entry)

            elif action_type == "name":
                entry = data['entry']
                channel_name = data['channel_name']
                num_channels = data['num_channels']
                self.undo_name_action(entry, channel_name, num_channels)

            elif action_type == "modify":
                self.undo_modify_action()
            
    def load_mapfile(self):
        # Opens a dialog to select the mapfile
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Mapfile", "", "Mapfile (*.map);;All Files (*)")

        if file_name:
            try:
                with open(file_name, 'r') as file:
                    lines = file.readlines()

                lines = lines[:-5]
                lines = [line[:-4] for line in lines]

                self.process_loaded_mapfile(lines)

            except Exception as e:
                self.generated_channels_text.setPlainText(f"Error loading the file: {str(e)}")

    def process_loaded_mapfile(self, lines):
        self.reset_generator()  # Reset existing data
        self.loaded_mapfile_lines = lines  # Save loaded lines

        for line in lines:
            line = line.strip('#')
            if line:
                parts = line.split(';')
                channel_info = f'{parts[0].strip()};'
                channel_name = parts[1].strip()
                if 'ref' in channel_name:
                    channel_name = channel_name.replace('ref', '')
                self.generated_channels.append(f'{channel_info} {channel_name}')
                self.channel_names[channel_info] = channel_name

        self.generated_content = self.generated_channels
        self.update_generated_channels_text()
        self.update_mapfile_text("Mapfile loaded.")

    def undo_generate_action(self, entry):
        if entry in self.generated_content:
            del self.generated_content[entry]
            self.num_ports.pop(entry, None)

        for generated_channels in reversed(self.generated_channels):
            if entry in generated_channels:
                self.generated_channels.remove(generated_channels)
                pass
        
        message = f"Port {entry} channels removed."
        self.update_mapfile_text(message)
        self.update_generated_channels_text()

    def update_generated_channels_text(self):
        generated_channels_text = "\n".join(self.generated_channels)
        self.generated_channels_text.setPlainText(generated_channels_text)

    def undo_name_action(self, entry, channel_name, num_channels):
        if entry in self.generated_content:
            channels = self.generated_content[entry]
            all_channels = [channel for port_channels in channels for channel in port_channels]

            num_named_channels_per_port = self.num_named_channels_per_port.get(entry, 0)
            if self.generate_18_channels_checkbox.isChecked():
                channels_per_port = 18
            else:
                channels_per_port = 32
            start_index = num_named_channels_per_port - num_channels  # Fixed index calculation
            end_index = start_index + num_channels

            for i in range(start_index, end_index):
                if 0 <= i < len(all_channels):
                    line, _ = all_channels[i]
                    all_channels[i] = (line, "")
                    self.channel_names.pop(line, None)  # Remove channel name

            for i, generated_channel in enumerate(self.generated_channels):
                if channel_name in generated_channel:
                    self.generated_channels[i] = self.generated_channels[i][:10]
                    pass

            self.num_named_channels_per_port[entry] = num_named_channels_per_port - num_channels
            self.names_preview.clear()
            self.update_generated_channels_text()
            message = f"Removed the {channel_name} name from {num_channels} channels on port {entry}."
            self.names_preview.setPlainText(message)

    def undo_modify_action(self):
        for line, channel_info in self.channel_names.items():
            name = channel_info
            suffix_option = self.combo_suffix_option.currentText()
            if '-Z' in name and suffix_option == "-Z":
                original_name = name.replace('-Z', '')
                self.channel_names[line] = original_name
                for i, generated_channel in enumerate(self.generated_channels):
                        if '-Z' in generated_channel:
                            self.generated_channels[i] = self.generated_channels[i][:-2]
                            pass
            elif '-COM' in name and suffix_option == "-COM":
                original_name = name.replace('-COM', '')
                self.channel_names[line] = original_name
                for i, generated_channel in enumerate(self.generated_channels):
                        if '-COM' in generated_channel:
                            self.generated_channels[i] = self.generated_channels[i][:-4]
                            pass
        self.modify_preview.clear()
        self.update_generated_channels_text()
        message = f"The {suffix_option} suffix has been removed from channel {original_name}."
        self.modify_preview.setPlainText(message)

    def update_mapfile_text(self,message=""):
        lines = []
        if message != "":
            lines.append(message)
        else:
            for entry, entry_content in self.generated_content.items():
                for port_content in entry_content:
                    for line, name in port_content:
                        if name:
                            line_with_name = f"{line} {name}"
                            lines.append(line_with_name)
                        else:
                            lines.append(line)

        content_text = "\n".join(lines)
        self.mapfile_text.setPlainText(content_text)

    def next_port(self):
        entry = self.combo_entry.currentText()
        if entry in self.num_ports:
            num_named_channels_per_port = self.num_named_channels_per_port.get(entry, 0)
            if self.generate_18_channels_checkbox.isChecked():
                channels_per_port = 18
            else:
                channels_per_port = 32

            if num_named_channels_per_port + channels_per_port <= self.num_ports[entry] * channels_per_port:
                next_port_number = (num_named_channels_per_port // channels_per_port) + 2
                self.num_named_channels_per_port[entry] = (next_port_number - 1) * channels_per_port
                self.names_preview.setPlainText(f"Starting naming at {entry}.{next_port_number:02d}.001")
            else:
                self.names_preview.setPlainText("All ports named.")
        else:
            self.names_preview.setPlainText("Generate channels for this entry first!")

    def name_channels(self):
        channel_name = self.line_channel_name.text()
        num_channels = int(self.line_num_channels.text())
        entry = self.combo_entry.currentText()

        if entry in self.generated_content:
            channels = self.generated_content[entry]
            all_channels = [channel for port_channels in channels for channel in port_channels]
            num_total_channels = sum(len(port_channels) for port_channels in channels)

            if num_channels > 0 and num_total_channels >= num_channels:
                names_preview_text = []

                if entry in self.num_named_channels_per_port:
                    num_named_channels_per_port = self.num_named_channels_per_port[entry]
                else:
                    num_named_channels_per_port = 0

                for i in range(1, num_channels + 1):
                    if num_named_channels_per_port + i <= len(all_channels):
                        line, _ = all_channels[num_named_channels_per_port + i - 1]
                        all_channels[num_named_channels_per_port + i - 1] = (line, f"{channel_name}{i:02d}")
                        names_preview_text.append(f"{line} {channel_name}{i:02d}")
                        self.channel_names[line] = f"{channel_name}{i:02d}"

                self.num_named_channels_per_port[entry] = num_named_channels_per_port + num_channels
                self.names_preview.setPlainText("\n".join(names_preview_text))

                for names_preview_text in names_preview_text:
                    for i, generated_channel in enumerate(self.generated_channels):
                        if generated_channel in names_preview_text:
                            self.generated_channels[i] = names_preview_text
                            pass

            else:
                error_message = "Cannot assign names to more channels than generated!"
                self.names_preview.setPlainText(error_message)
        else:
            error_message = "Generate channels for this entry first!"
            self.names_preview.setPlainText(error_message)

        self.update_generated_channels_text()
        action_data = {"entry": entry, "channel_name": channel_name, "num_channels": num_channels}
        self.record_action(("name", action_data))

    def modify_channel_name(self):
        modify_text = self.line_modify_channel.text()
        suffix_option = self.combo_suffix_option.currentText()
        entry = self.combo_entry.currentText()

        if modify_text and entry in self.generated_content:
            modified_channels = 0
            success_message = ""

            for i, (line, name) in enumerate(self.channel_names.items()):
                if modify_text in name:
                    modified_channels += 1

                    if suffix_option == "-Z":
                        self.channel_names[line] = f"{name}-Z"
                        for i, generated_channel in enumerate(self.generated_channels):
                            if '-Z' in generated_channel:
                                self.generated_channels[i] = self.generated_channels[i][:-2]
                                pass

                            if modify_text in generated_channel:
                                self.generated_channels[i] = f"{self.generated_channels[i]}-Z"
                                pass
                        success_message = f"ground"
                    elif suffix_option == "-COM":
                        self.channel_names[line] = f"{name}-COM"
                        for i, generated_channel in enumerate(self.generated_channels):
                            if '-COM' in generated_channel:
                                self.generated_channels[i] = self.generated_channels[i][:-4]
                                pass
                            if modify_text in generated_channel:
                                self.generated_channels[i] = f"{self.generated_channels[i]}-COM"
                                pass
                        success_message = f"reference"

            if modified_channels > 0:
                self.update_generated_channels_text()
                if success_message:
                    message = f"Added the {success_message} suffix to {modified_channels} channel named {modify_text}."
            else:
                message = "No channels found with the specified text in name preview."
        else:
            message = "Enter text and select an existing entry."

        self.modify_preview.setPlainText(message)

        action_data = {"modified_channels": modified_channels}
        self.record_action(("modify", action_data))

    def save_mapfile(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save mapfile", "", "Mapfile (*.map);;All Files (*)")
        if file_name:
            with open(file_name, 'w') as file:
                lines = []
                prev_entry = None
                prev_port = None

                for entry, entry_content in self.generated_content.items():
                    for port_channels in entry_content:
                        for line, _ in port_channels:
                            if prev_entry is not None and prev_entry != entry:
                                lines.append("")  # Add a blank line when changing entry
                            if prev_port is not None and prev_port != line[4:7]:
                                lines.append("")  # Add a blank line when changing port
                            if line in self.channel_names:
                                name = self.channel_names[line]
                                if 'm' in name and '09' in name:
                                    name += "ref"
                                line_with_name = f"{line} {name}; ;"
                            else:
                                line_with_name = f"{line}; ;"
                            lines.append(line_with_name)
                            prev_entry = entry
                            prev_port = line[4:7]

                # Add additional lines at the end of the file
                lines.append("")
                lines.append("1.DIO.BNC.001; Photo_digital; ;")
                lines.append("1.DIO.PAR.001; parallel_dig; ;")
                lines.append("1.AIO.AUD.001; MicL; ;")
                lines.append("1.AIO.AUD.002; MicR; ;")
                lines.append("1.AIO.BNC.001; Photo_analog; ;")
                lines.append("1.AIO.BNC.002; Mic2; ;")

                # Comment lines with 15 or fewer characters
                for i, line in enumerate(lines):
                    if len(line) <= 15:
                        lines[i] = f"# {line}"

                file.write("\n".join(lines))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MapfileGenerator()
    window.show()
    sys.exit(app.exec())