"""Bounded direct parameter request; no ROS CLI daemon or unbounded spin."""
import argparse,os,sys
import rclpy
from rcl_interfaces.msg import Parameter,ParameterValue,ParameterType
from rcl_interfaces.srv import GetParameters,SetParameters


def bounded_call(node,client,request,timeout=3.):
    if not client.wait_for_service(timeout_sec=timeout):raise RuntimeError('列参数服务未就绪')
    future=client.call_async(request)
    rclpy.spin_until_future_complete(node,future,timeout_sec=timeout)
    if not future.done():
        future.cancel()
        raise RuntimeError('列参数请求超时')
    result=future.result()
    if result is None:raise RuntimeError('列参数响应为空')
    return result


def set_column(node,column):
    if column not in range(5):raise ValueError('列号必须为0..4')
    setter=node.create_client(SetParameters,'/smolvla_grasp_assistant/set_parameters')
    getter=node.create_client(GetParameters,'/smolvla_grasp_assistant/get_parameters')
    request=SetParameters.Request(parameters=[Parameter(name='cup_collection_column',value=ParameterValue(type=ParameterType.PARAMETER_INTEGER,integer_value=column))])
    result=bounded_call(node,setter,request)
    if len(result.results)!=1 or not result.results[0].successful:raise RuntimeError('列参数设置失败')
    reply=bounded_call(node,getter,GetParameters.Request(names=['cup_collection_column']))
    if (len(reply.values)!=1 or reply.values[0].type!=ParameterType.PARAMETER_INTEGER
            or reply.values[0].integer_value!=column):raise RuntimeError('列参数回读不一致')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('column',type=int,choices=range(5));args=parser.parse_args()
    rclpy.init();node=rclpy.create_node('cup_column_setter_'+str(os.getpid()))
    try:
        set_column(node,args.column)
        print('杯子列号设置并回读确认：'+str(args.column),flush=True)
    except Exception as exc:
        print(str(exc),file=sys.stderr,flush=True);return 1
    finally:
        node.destroy_node();rclpy.shutdown()
    return 0

if __name__=='__main__':raise SystemExit(main())
