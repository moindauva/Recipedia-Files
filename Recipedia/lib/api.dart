import 'dart:convert' as convert;
import 'dart:convert';
import 'package:http/http.dart' as http;

Future<List<dynamic>> API(var base64Image, context) async {
  var url = Uri.http('192.168.0.106:5000', '/api/detect');
  Map<String, String> data = {'img': base64Image};
  Map<String, String> headers = {"Content-Type": "application/json","Accept": "application/json"};
  var response = await http.post(url,body:json.encode(data),headers:headers);
  print('Response Code: ${response.statusCode}');
  if (response.statusCode == 200) {
    var jsonResponse =
    convert.jsonDecode(response.body) as Map<String, dynamic>;
    List<dynamic> output = jsonResponse['data'];
    print('Items scanned are: $output.');
    return output;
  } else {
    //return 'Error occurred try-again!';
    return [];
  }
}