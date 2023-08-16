import 'package:flutter/material.dart';

class About extends StatelessWidget {
  const About({super.key});

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Scaffold(
      appBar: AppBar(
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.only(
              bottomRight: Radius.circular(25),
              bottomLeft: Radius.circular(25)),
        ),
        centerTitle: true,
        title: const Text(
          'About Us',
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              /*SizedBox(
                height: size.height * 0.03,
              ),
              const Text(
                "About Us",
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                ),
              ),*/
              SizedBox(
                height: size.height * 0.02,
              ),
              Column(
                mainAxisAlignment: MainAxisAlignment.start,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    textAlign: TextAlign.left,
                    'Recipedia',
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(
                    height: size.height * 0.02,
                  ),
                  TextField(
                    enabled: false,
                    // Make the text field read-only
                    keyboardType: TextInputType.multiline,
                    maxLines: null,
                    // Allows unlimited lines
                    decoration: const InputDecoration(
                      border: InputBorder.none,
                    ),
                    style: const TextStyle(color: Colors.black87),
                    controller: TextEditingController(
                        text:
                            'Recipedia is a user-friendly platform targeting housewives and bachelors who don’t know how to cook and also who are interested. The user will scan the vegetables or fruits, and the app will suggest the recipes for foods or juices according to the scan vegetables or fruits. App also includes an add-to-favorite feature in which the user can add any recipe to that favorite list. There will be a feature for sharing a recipe with your friends through social media. If the user likes or dislikes the recipe, they can rate the recipe. There will be an admin portal from where the admin can use CRUD operation for recipes.'), // Set the initial value
                  ),
                ],
              ),
              SizedBox(
                height: size.height * 0.03,
              ),
              const Text(
                'Developers',
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                ),
              ),
              SizedBox(
                height: size.height * 0.03,
              ),
              Column(
                mainAxisAlignment: MainAxisAlignment.start,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    textAlign: TextAlign.left,
                    'Safin Mahesania',
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: size.height * 0.02),
                  Row(
                    children: [
                      SizedBox(
                        width: size.width * 0.3,
                        child: Image.asset(
                          'assets/safin.jpeg',
                        ),
                      ),
                      SizedBox(
                        width: size.width * 0.598,
                        child: TextField(
                          enabled: false,
                          // Make the text field read-only
                          keyboardType: TextInputType.multiline,
                          maxLines: null,
                          // Allows unlimited lines
                          decoration: const InputDecoration(
                            contentPadding: EdgeInsets.only(left: 16.0),
                            border: InputBorder.none,
                          ),
                          style: const TextStyle(color: Colors.black87),
                          controller: TextEditingController(
                              text:
                                  "Experienced backend developer skilled in server-side programming, Python, and relevant frameworks. I excel in writing efficient, scalable, and robust code to power backend infrastructure. My expertise includes database design, API integrations, and data processing. I troubleshoot and optimize performance for seamless operation and user experience. I stay updated with emerging technologies."), // Set the initial value
                        ),
                      ),
                    ],
                  ),
                  /*Row(
                    children: [
                      Image.asset('assets/safin.jpeg', height: 200),
                      //SizedBox(height: size.height * 0.02),
                      Container(
                        /*decoration: BoxDecoration(
                          border: Border.all(
                            color: Colors.grey,
                            width: 2.0,
                          ),
                          borderRadius: BorderRadius.circular(8.0),
                        ),*/
                        child: TextField(
                          enabled: false,
                          // Make the text field read-only
                          keyboardType: TextInputType.multiline,
                          maxLines: null,
                          // Allows unlimited lines
                          decoration: InputDecoration(
                            contentPadding: EdgeInsets.all(16.0),
                            border: InputBorder.none,
                          ),
                          controller: TextEditingController(
                              text:
                                  'As a highly skilled backend developer, I possess a deep understanding of server-side programming and have a proven track record in building and maintaining the essential functionality of applications and websites. With expertise in languages like Python and other relevant frameworks, I excel in writing efficient, scalable, and robust code that powers the backend infrastructure. I am experienced in designing and implementing databases, API integrations, and handling data processing tasks. My strong problem-solving abilities enable me to troubleshoot and optimize performance, ensuring seamless operation and excellent user experience. Continuously staying updated with emerging technologies, I am committed to delivering secure, high-performing, and scalable backend solutions that meet the specific needs of the project.'), // Set the initial value
                        ),
                      ),
                    ],
                  ),*/
                  SizedBox(height: size.height * 0.02),
                  const Text(
                    textAlign: TextAlign.left,
                    'Moin Dauva',
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: size.height * 0.02),
                  Row(
                    children: [
                      SizedBox(
                        width: size.width * 0.598,
                        child: TextField(
                          enabled: false,
                          // Make the text field read-only
                          keyboardType: TextInputType.multiline,
                          maxLines: null,
                          style: const TextStyle(color: Colors.black87),
                          // Allows unlimited lines
                          decoration: const InputDecoration(
                            contentPadding: EdgeInsets.only(right: 16.0),
                            border: InputBorder.none,
                          ),
                          controller: TextEditingController(
                              text:
                                  "Passionate frontend developer skilled in mobile and desktop web development. Creates dynamic, responsive, and user-friendly applications. Builds intuitive interfaces for seamless cross-platform experiences. Proficient in HTML, CSS, JavaScript, and related frameworks. Committed to enhancing functionality and aesthetics through continuous learning and adoption of new technologies."),
                        ),
                      ),
                      SizedBox(
                        width: size.width * 0.3,
                        child: Image.asset(
                          'assets/moin.jpg',
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
