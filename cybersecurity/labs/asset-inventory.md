<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Home Asset Inventory</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 25px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        th {
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        td {
            border: 1px solid #ddd;
            padding: 10px;
            vertical-align: top;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .sensitivity {
            font-weight: bold;
            padding: 3px 8px;
            border-radius: 3px;
        }
        .restricted {
            background-color: #ffcccc;
            color: #990000;
        }
        .confidential {
            background-color: #ffe6cc;
            color: #cc6600;
        }
        .internal {
            background-color: #ffffe6;
            color: #666600;
        }
        .public {
            background-color: #e6ffe6;
            color: #006600;
        }
        .categories-table {
            width: 50%;
            margin-top: 20px;
        }
        .note {
            background-color: #e3f2fd;
            padding: 15px;
            border-left: 4px solid #2196F3;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>Home Office Network Asset Inventory</h1>
    
    <div class="note">
        <strong>Scenario:</strong> Small business home office network inventory for security assessment
    </div>

    <h2>Network Device Inventory</h2>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Asset</th>
                <th>Network Access</th>
                <th>Owner</th>
                <th>Location</th>
                <th>Notes</th>
                <th>Sensitivity</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>1</td>
                <td><strong>Network router</strong></td>
                <td>Continuous</td>
                <td>Internet service provider (ISP)</td>
                <td>On-premises</td>
                <td>Has a 2.4 GHz and 5 GHz connection. All devices on the home network connect to the 5 GHz frequency.</td>
                <td><span class="sensitivity confidential">Confidential</span></td>
            </tr>
            <tr>
                <td>2</td>
                <td><strong>Desktop</strong></td>
                <td>Occasional</td>
                <td>Homeowner</td>
                <td>On-premises</td>
                <td>Contains private information, like photos.</td>
                <td><span class="sensitivity restricted">Restricted</span></td>
            </tr>
            <tr>
                <td>3</td>
                <td><strong>Guest smartphone</strong></td>
                <td>Occasional</td>
                <td>Friend</td>
                <td>On and Off-premises</td>
                <td>Connects to my home network.</td>
                <td><span class="sensitivity internal">Internal-only</span></td>
            </tr>
            <tr style="background-color: #fff3e0;">
                <td>4</td>
                <td><strong>External hard drive</strong></td>
                <td>Occasional</td>
                <td>Homeowner</td>
                <td>On-premises (home office)</td>
                <td>Contains business backups and sensitive financial records. Connected via USB to desktop, network-accessible for backups.</td>
                <td><span class="sensitivity restricted">Restricted</span></td>
            </tr>
            <tr style="background-color: #fff3e0;">
                <td>5</td>
                <td><strong>Wireless printer</strong></td>
                <td>Continuous</td>
                <td>Homeowner</td>
                <td>On-premises (home office)</td>
                <td>Network-enabled printer accessible by all home devices. May store print jobs temporarily in memory.</td>
                <td><span class="sensitivity confidential">Confidential</span></td>
            </tr>
            <tr style="background-color: #fff3e0;">
                <td>6</td>
                <td><strong>Smart TV/Streaming device</strong></td>
                <td>Continuous</td>
                <td>Homeowner</td>
                <td>On-premises (living room)</td>
                <td>Connected to 5GHz network. Has access to streaming accounts and browsing history.</td>
                <td><span class="sensitivity internal">Internal-only</span></td>
            </tr>
        </tbody>
    </table>

    <h2>Sensitivity Classification Categories</h2>
    <table class="categories-table">
        <thead>
            <tr>
                <th>Category</th>
                <th>Access Designation</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><span class="sensitivity restricted">Restricted</span></td>
                <td>Need-to-know basis only</td>
            </tr>
            <tr>
                <td><span class="sensitivity confidential">Confidential</span></td>
                <td>Limited to specific users</td>
            </tr>
            <tr>
                <td><span class="sensitivity internal">Internal-only</span></td>
                <td>Users on-premises</td>
            </tr>
            <tr>
                <td><span class="sensitivity public">Public</span></td><!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Home Asset Inventory</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 25px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        th {
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        td {
            border: 1px solid #ddd;
            padding: 10px;
            vertical-align: top;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .sensitivity {
            font-weight: bold;
            padding: 3px 8px;
            border-radius: 3px;
        }
        .restricted {
            background-color: #ffcccc;
            color: #990000;
        }
        .confidential {
            background-color: #ffe6cc;
            color: #cc6600;
        }
        .internal {
            background-color: #ffffe6;
            color: #666600;
        }
        .public {
            background-color: #e6ffe6;
            color: #006600;
        }
        .categories-table {
            width: 50%;
            margin-top: 20px;
        }
        .note {
            background-color: #e3f2fd;
            padding: 15px;
            border-left: 4px solid #2196F3;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>Home Office Network Asset Inventory</h1>
    
    <div class="note">
        <strong>Scenario:</strong> Small business home office network inventory for security assessment
    </div>

    <h2>Network Device Inventory</h2>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Asset</th>
                <th>Network Access</th>
                <th>Owner</th>
                <th>Location</th>
                <th>Notes</th>
                <th>Sensitivity</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>1</td>
                <td><strong>Network router</strong></td>
                <td>Continuous</td>
                <td>Internet service provider (ISP)</td>
                <td>On-premises</td>
                <td>Has a 2.4 GHz and 5 GHz connection. All devices on the home network connect to the 5 GHz frequency.</td>
                <td><span class="sensitivity confidential">Confidential</span></td>
            </tr>
            <tr>
                <td>2</td>
                <td><strong>Desktop</strong></td>
                <td>Occasional</td>
                <td>Homeowner</td>
                <td>On-premises</td>
                <td>Contains private information, like photos.</td>
                <td><span class="sensitivity restricted">Restricted</span></td>
            </tr>
            <tr>
                <td>3</td>
                <td><strong>Guest smartphone</strong></td>
                <td>Occasional</td>
                <td>Friend</td>
                <td>On and Off-premises</td>
                <td>Connects to my home network.</td>
                <td><span class="sensitivity internal">Internal-only</span></td>
            </tr>
            <tr style="background-color: #fff3e0;">
                <td>4</td>
                <td><strong>External hard drive</strong></td>
                <td>Occasional</td>
                <td>Homeowner</td>
                <td>On-premises (home office)</td>
                <td>Contains business backups and sensitive financial records. Connected via USB to desktop, network-accessible for backups.</td>
                <td><span class="sensitivity restricted">Restricted</span></td>
            </tr>
            <tr style="background-color: #fff3e0;">
                <td>5</td>
                <td><strong>Wireless printer</strong></td>
                <td>Continuous</td>
                <td>Homeowner</td>
                <td>On-premises (home office)</td>
                <td>Network-enabled printer accessible by all home devices. May store print jobs temporarily in memory.</td>
                <td><span class="sensitivity confidential">Confidential</span></td>
            </tr>
            <tr style="background-color: #fff3e0;">
                <td>6</td>
                <td><strong>Smart TV/Streaming device</strong></td>
                <td>Continuous</td>
                <td>Homeowner</td>
                <td>On-premises (living room)</td>
                <td>Connected to 5GHz network. Has access to streaming accounts and browsing history.</td>
                <td><span class="sensitivity internal">Internal-only</span></td>
            </tr>
        </tbody>
    </table>

    <h2>Sensitivity Classification Categories</h2>
    <table class="categories-table">
        <thead>
            <tr>
                <th>Category</th>
                <th>Access Designation</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><span class="sensitivity restricted">Restricted</span></td>
                <td>Need-to-know basis only</td>
            </tr>
            <tr>
                <td><span class="sensitivity confidential">Confidential</span></td>
                <td>Limited to specific users</td>
            </tr>
            <tr>
                <td><span class="sensitivity internal">Internal-only</span></td>
                <td>Users on-premises</td>
            </tr>
            <tr>
                <td><span class="sensitivity public">Public</span></td><!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Home Asset Inventory</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 25px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        th {
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        td {
            border: 1px solid #ddd;
            padding: 10px;
            vertical-align: top;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .sensitivity {
            font-weight: bold;
            padding: 3px 8px;
            border-radius: 3px;
        }
        .restricted {
            background-color: #ffcccc;
            color: #990000;
        }
        .confidential {
            background-color: #ffe6cc;
            color: #cc6600;
        }
        .internal {
            background-color: #ffffe6;
            color: #666600;
        }
        .public {
            background-color: #e6ffe6;
            color: #006600;
        }
        .categories-table {
            width: 50%;
            margin-top: 20px;
        }
        .note {
            background-color: #e3f2fd;
            padding: 15px;
            border-left: 4px solid #2196F3;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>Home Office Network Asset Inventory</h1>
    
    <div class="note">
        <strong>Scenario:</strong> Small business home office network inventory for security assessment
    </div>

    <h2>Network Device Inventory</h2>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Asset</th>
                <th>Network Access</th>
                <th>Owner</th>
                <th>Location</th>
                <th>Notes</th>
                <th>Sensitivity</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>1</td>
                <td><strong>Network router</strong></td>
                <td>Continuous</td>
                <td>Internet service provider (ISP)</td>
                <td>On-premises</td>
                <td>Has a 2.4 GHz and 5 GHz connection. All devices on the home network connect to the 5 GHz frequency.</td>
                <td><span class="sensitivity confidential">Confidential</span></td>
            </tr>
            <tr>
                <td>2</td>
                <td><strong>Desktop</strong></td>
                <td>Occasional</td>
                <td>Homeowner</td>
                <td>On-premises</td>
                <td>Contains private information, like photos.</td>
                <td><span class="sensitivity restricted">Restricted</span></td>
            </tr>
            <tr>
                <td>3</td>
                <td><strong>Guest smartphone</strong></td>
                <td>Occasional</td>
                <td>Friend</td>
                <td>On and Off-premises</td>
                <td>Connects to my home network.</td>
                <td><span class="sensitivity internal">Internal-only</span></td>
            </tr>
            <tr style="background-color: #fff3e0;">
                <td>4</td>
                <td><strong>External hard drive</strong></td>
                <td>Occasional</td>
                <td>Homeowner</td>
                <td>On-premises (home office)</td>
                <td>Contains business backups and sensitive financial records. Connected via USB to desktop, network-accessible for backups.</td>
                <td><span class="sensitivity restricted">Restricted</span></td>
            </tr>
            <tr style="background-color: #fff3e0;">
                <td>5</td>
                <td><strong>Wireless printer</strong></td>
                <td>Continuous</td>
                <td>Homeowner</td>
                <td>On-premises (home office)</td>
                <td>Network-enabled printer accessible by all home devices. May store print jobs temporarily in memory.</td>
                <td><span class="sensitivity confidential">Confidential</span></td>
            </tr>
            <tr style="background-color: #fff3e0;">
                <td>6</td>
                <td><strong>Smart TV/Streaming device</strong></td>
                <td>Continuous</td>
                <td>Homeowner</td>
                <td>On-premises (living room)</td>
                <td>Connected to 5GHz network. Has access to streaming accounts and browsing history.</td>
                <td><span class="sensitivity internal">Internal-only</span></td>
            </tr>
        </tbody>
    </table>

    <h2>Sensitivity Classification Categories</h2>
    <table class="categories-table">
        <thead>
            <tr>
                <th>Category</th>
                <th>Access Designation</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><span class="sensitivity restricted">Restricted</span></td>
                <td>Need-to-know basis only</td>
            </tr>
            <tr>
                <td><span class="sensitivity confidential">Confidential</span></td>
                <td>Limited to specific users</td>
            </tr>
            <tr>
                <td><span class="sensitivity internal">Internal-only</span></td>
                <td>Users on-premises</td>
            </tr>
            <tr>
                <td><span class="sensitivity public">Public</span></td>
                <td>Anyone</td>
            </tr>
        </tbody>
    </table>

    <h2>Justification for New Devices</h2>
    <ul>
        <li><strong>External Hard Drive:</strong> Classified as <em>Restricted</em> because it contains critical business backups and financial records that should only be accessed on a need-to-know basis.</li>
        <li><strong>Wireless Printer:</strong> Classified as <em>Confidential</em> because it can store print jobs in memory and should have limited user access to prevent unauthorized document viewing.</li>
        <li><strong>Smart TV/Streaming Device:</strong> Classified as <em>Internal-only</em> because it contains personal streaming accounts and browsing history, accessible to household members.</li>
    </ul>
</body>
</html>
                <td>Anyone</td>
            </tr>
        </tbody>
    </table>

    <h2>Justification for New Devices</h2>
    <ul>
        <li><strong>External Hard Drive:</strong> Classified as <em>Restricted</em> because it contains critical business backups and financial records that should only be accessed on a need-to-know basis.</li>
        <li><strong>Wireless Printer:</strong> Classified as <em>Confidential</em> because it can store print jobs in memory and should have limited user access to prevent unauthorized document viewing.</li>
        <li><strong>Smart TV/Streaming Device:</strong> Classified as <em>Internal-only</em> because it contains personal streaming accounts and browsing history, accessible to household members.</li>
    </ul>
</body>
</html>
                <td>Anyone</td>
            </tr>
        </tbody>
    </table>

    <h2>Justification for New Devices</h2>
    <ul>
        <li><strong>External Hard Drive:</strong> Classified as <em>Restricted</em> because it contains critical business backups and financial records that should only be accessed on a need-to-know basis.</li>
        <li><strong>Wireless Printer:</strong> Classified as <em>Confidential</em> because it can store print jobs in memory and should have limited user access to prevent unauthorized document viewing.</li>
        <li><strong>Smart TV/Streaming Device:</strong> Classified as <em>Internal-only</em> because it contains personal streaming accounts and browsing history, accessible to household members.</li>
    </ul>
</body>
</html>