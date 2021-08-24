# EmailTools
Outlook 宏脚本内容  
```vba
Private Sub Application_NewMail()
    Shell "python <script_path>\EmailTools.py"
End Sub
```