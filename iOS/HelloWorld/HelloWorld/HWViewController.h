//
//  HWViewController.h
//  HelloWorld
//
//  Created by Shajan Dasan on 1/22/14.
//  Copyright (c) 2014 Shajan Dasan. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface HWViewController : UIViewController
@property (weak, nonatomic) IBOutlet UILabel *label;
@property (weak, nonatomic) IBOutlet UITextField *text;
- (IBAction)button:(id)sender;
@end
